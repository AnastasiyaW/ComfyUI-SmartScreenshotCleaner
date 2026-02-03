import torch
import numpy as np
from typing import Tuple, List, Optional
import os
import logging

# Настраиваем логгер
logger = logging.getLogger("SmartScreenshotCleaner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_device() -> torch.device:
    """Определяет доступное устройство."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AutoBorderCrop:
    """
    Автоматически обрезает однотонные рамки любого цвета.
    Поддерживает batch processing.
    """

    # Константы для детекции
    SENSITIVITY_MULTIPLIER = 2.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": ("FLOAT", {
                    "default": 15.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "slider",
                    "tooltip": "Чувствительность детекции рамки (меньше = строже)"
                }),
                "min_border_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Минимальный размер рамки в пикселях"
                }),
                "min_content_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 10.0,
                    "max": 90.0,
                    "step": 5.0,
                    "tooltip": "Минимальный процент контента после обрезки"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("image", "crop_top", "crop_bottom", "crop_left", "crop_right", "was_cropped")
    FUNCTION = "crop_borders"
    CATEGORY = "image/preprocessing"

    def crop_borders(
        self,
        image: torch.Tensor,
        sensitivity: float = 15.0,
        min_border_size: int = 5,
        min_content_percent: float = 50.0,
    ) -> Tuple[torch.Tensor, int, int, int, int, bool]:
        """Обрезает рамки. Поддерживает batch."""

        batch_size = image.shape[0]
        results = []
        crop_values = None

        for b in range(batch_size):
            img = image[b].cpu().numpy()
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1

            # Убедимся что есть 3 канала
            if c == 1:
                img = np.stack([img.squeeze()] * 3, axis=-1)

            img_uint8 = (img * 255).astype(np.uint8)

            crop_top = self._detect_border(img_uint8, 'top', sensitivity, min_border_size)
            crop_bottom = self._detect_border(img_uint8, 'bottom', sensitivity, min_border_size)
            crop_left = self._detect_border(img_uint8, 'left', sensitivity, min_border_size)
            crop_right = self._detect_border(img_uint8, 'right', sensitivity, min_border_size)

            # Валидация
            remaining_h = h - crop_top - crop_bottom
            remaining_w = w - crop_left - crop_right
            min_h = int(h * min_content_percent / 100)
            min_w = int(w * min_content_percent / 100)

            if remaining_h < min_h or remaining_w < min_w or remaining_h <= 0 or remaining_w <= 0:
                crop_top, crop_bottom, crop_left, crop_right = 0, 0, 0, 0

            # Сохраняем crop values от первого изображения
            if crop_values is None:
                crop_values = (crop_top, crop_bottom, crop_left, crop_right)

            # Применяем обрезку
            y_end = h - crop_bottom if crop_bottom > 0 else h
            x_end = w - crop_right if crop_right > 0 else w
            cropped = image[b:b+1, crop_top:y_end, crop_left:x_end, :]
            results.append(cropped)

        # Собираем batch
        output = torch.cat(results, dim=0) if results else image
        was_cropped = any(v > 0 for v in crop_values) if crop_values else False

        return (output, *crop_values, was_cropped)

    def _detect_border(self, img: np.ndarray, side: str, sensitivity: float, min_border_size: int) -> int:
        """Детектирует границу рамки с указанной стороны."""
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1

        # Настройки для каждой стороны
        if side == 'top':
            get_line = lambda i: img[i, :, :3]
            max_scan = h // 2
        elif side == 'bottom':
            get_line = lambda i: img[h - 1 - i, :, :3]
            max_scan = h // 2
        elif side == 'left':
            get_line = lambda i: img[:, i, :3]
            max_scan = w // 2
        else:  # right
            get_line = lambda i: img[:, w - 1 - i, :3]
            max_scan = w // 2

        first_line = get_line(0).astype(np.float32)
        reference_color = np.median(first_line, axis=0)
        first_line_std = np.std(first_line, axis=0).mean()

        if first_line_std > sensitivity * self.SENSITIVITY_MULTIPLIER:
            return 0

        border_end = 0
        for i in range(max_scan):
            line = get_line(i).astype(np.float32)
            line_avg = np.mean(line, axis=0)
            color_diff = np.abs(line_avg - reference_color).mean()
            line_std = np.std(line, axis=0).mean()

            if color_diff < sensitivity and line_std < sensitivity * self.SENSITIVITY_MULTIPLIER:
                border_end = i + 1
            else:
                break

        return border_end if border_end >= min_border_size else 0


class SmartScreenshotCleaner:
    """
    Умная нода для очистки скриншотов от UI элементов.

    Возможности:
    - YOLO детекция UI (deki-yolo) - GPU accelerated
    - SOD детекция контента (U2Net) - GPU accelerated
    - LaMa инпеинтинг - GPU accelerated
    - Batch processing

    Модели скачиваются автоматически при первом запуске.
    """

    # Константы моделей
    MODEL_REPO_YOLO = "orasul/deki-yolo"
    MODEL_FILE_YOLO = "best.pt"

    # Константы для анализа
    EDGE_SAMPLE_SIZE = 20
    DARK_UI_THRESHOLD = 60
    BRIGHTNESS_STD_THRESHOLD = 20
    DARK_BRIGHTNESS_MAX = 70
    LIGHT_BRIGHTNESS_MIN = 200
    UI_DENSITY_THRESHOLD = 0.2

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["hybrid", "yolo_only", "sod_only", "full_inpaint"], {
                    "default": "hybrid",
                    "tooltip": "hybrid=YOLO+SOD, yolo_only=только UI детекция, sod_only=только контент, full_inpaint=очистка+заливка"
                }),
                "yolo_confidence": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Порог уверенности YOLO"
                }),
                "sod_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Порог SOD маски"
                }),
                "expand_mask_px": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Расширение маски UI для инпеинтинга"
                }),
                "crop_to_content": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Обрезать до контента"
                }),
                "min_content_ratio": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.1,
                    "max": 0.8,
                    "step": 0.05,
                    "tooltip": "Минимальная доля контента"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cleaned_image", "original_cropped", "ui_mask", "content_mask",
                    "crop_top", "crop_bottom", "crop_left", "crop_right")
    FUNCTION = "process"
    CATEGORY = "image/preprocessing"

    def __init__(self):
        self.device = get_device()
        self.yolo_model = None
        self.sod_session = None
        self.lama_model = None
        self._models_dir = self._get_models_dir()
        logger.info(f"Using device: {self.device}")

    def _get_models_dir(self) -> str:
        """Получает директорию для моделей."""
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "screenshot_cleaner")
        except ImportError:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        os.makedirs(models_dir, exist_ok=True)
        return models_dir

    # ==================== YOLO (GPU) ====================

    def _load_yolo(self):
        """Загружает YOLO модель на GPU."""
        if self.yolo_model is not None:
            return self.yolo_model

        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download

            logger.info(f"Downloading YOLO model from {self.MODEL_REPO_YOLO}...")

            model_path = hf_hub_download(
                repo_id=self.MODEL_REPO_YOLO,
                filename=self.MODEL_FILE_YOLO,
                cache_dir=self._models_dir
            )

            self.yolo_model = YOLO(model_path)

            # Переносим на GPU если доступно
            if self.device.type == "cuda":
                self.yolo_model.to(self.device)

            logger.info(f"YOLO loaded on {self.device}")
            return self.yolo_model

        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Install: pip install ultralytics huggingface_hub")
            return None
        except Exception as e:
            logger.error(f"YOLO load error: {e}")
            return None

    @torch.no_grad()
    def _detect_ui_yolo(self, img_np: np.ndarray, conf: float) -> Tuple[List[dict], np.ndarray]:
        """Детектирует UI элементы через YOLO на GPU."""
        model = self._load_yolo()
        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        if model is None:
            return [], mask

        # YOLO inference (автоматически на GPU если модель там)
        results = model.predict(img_np, conf=conf, verbose=False, device=self.device)
        detections = []

        class_names = ["View", "ImageView", "Text", "Line"]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().astype(int)
                conf_score = float(boxes.conf[i].cpu().item())
                cls = int(boxes.cls[i].cpu().item())

                class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"

                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": conf_score,
                    "class": class_name,
                })

                mask[y1:y2, x1:x2] = 1.0

        logger.info(f"YOLO detected {len(detections)} UI elements")
        return detections, mask

    # ==================== SOD (GPU) ====================

    def _load_sod(self):
        """Загружает SOD модель."""
        if self.sod_session is not None:
            return self.sod_session

        try:
            from rembg import new_session

            # Используем u2net с GPU провайдером если доступно
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == "cuda" else ['CPUExecutionProvider']

            logger.info(f"Loading SOD model with providers: {providers}")
            self.sod_session = new_session("u2net", providers=providers)
            logger.info("SOD loaded: rembg/u2net")
            return self.sod_session

        except ImportError as e:
            logger.error(f"rembg not found: {e}. Install: pip install rembg[gpu]")
            return None
        except Exception as e:
            logger.error(f"SOD load error: {e}")
            return None

    @torch.no_grad()
    def _detect_content_sod(self, img_np: np.ndarray, threshold: float) -> np.ndarray:
        """Детектирует контент через SOD."""
        h, w = img_np.shape[:2]

        session = self._load_sod()
        if session is None:
            return np.ones((h, w), dtype=np.float32)

        try:
            from PIL import Image
            from rembg import remove

            img_pil = Image.fromarray(img_np)
            result = remove(img_pil, session=session, only_mask=True)
            mask = np.array(result).astype(np.float32) / 255.0

            mask_binary = (mask > threshold).astype(np.float32)
            logger.info(f"SOD content coverage: {mask_binary.mean()*100:.1f}%")
            return mask_binary

        except Exception as e:
            logger.error(f"SOD error: {e}")
            return np.ones((h, w), dtype=np.float32)

    # ==================== LaMa Inpainting (GPU) ====================

    def _load_lama(self):
        """Загружает LaMa модель."""
        if self.lama_model is not None:
            return self.lama_model

        try:
            from simple_lama_inpainting import SimpleLama

            # SimpleLama автоматически использует CUDA если доступно
            self.lama_model = SimpleLama()
            logger.info(f"LaMa loaded (device: {'cuda' if torch.cuda.is_available() else 'cpu'})")
            return self.lama_model

        except ImportError as e:
            logger.warning(f"LaMa not available: {e}. Using OpenCV fallback.")
            return None
        except Exception as e:
            logger.error(f"LaMa load error: {e}")
            return None

    @torch.no_grad()
    def _inpaint_ui(self, img_np: np.ndarray, mask: np.ndarray, expand_px: int) -> np.ndarray:
        """Закрашивает UI элементы."""
        import cv2
        from PIL import Image

        if mask.max() == 0:
            return img_np

        # Расширяем маску
        if expand_px > 0:
            kernel = np.ones((expand_px * 2 + 1, expand_px * 2 + 1), np.uint8)
            mask_expanded = cv2.dilate((mask * 255).astype(np.uint8), kernel, iterations=1)
        else:
            mask_expanded = (mask * 255).astype(np.uint8)

        lama = self._load_lama()

        if lama is not None:
            try:
                img_pil = Image.fromarray(img_np)
                mask_pil = Image.fromarray(mask_expanded)
                result = lama(img_pil, mask_pil)
                return np.array(result)
            except Exception as e:
                logger.warning(f"LaMa inpaint error: {e}, falling back to OpenCV")

        # Fallback: OpenCV TELEA
        try:
            result = cv2.inpaint(img_np, mask_expanded, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            logger.info("Used OpenCV TELEA inpainting")
            return result
        except Exception as e:
            logger.error(f"OpenCV inpaint error: {e}")
            return img_np

    # ==================== Content Region Detection ====================

    def _find_content_bounds(
        self,
        h: int, w: int,
        ui_mask: np.ndarray,
        content_mask: np.ndarray,
        min_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Находит границы контента по маскам."""
        # Контент = SOD маска минус UI маска
        combined = content_mask * (1 - ui_mask)

        rows_with_content = np.any(combined > 0.5, axis=1)
        cols_with_content = np.any(combined > 0.5, axis=0)

        if not np.any(rows_with_content) or not np.any(cols_with_content):
            return (0, 0, 0, 0)

        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]

        top = int(row_indices[0])
        bottom = int(h - row_indices[-1] - 1)
        left = int(col_indices[0])
        right = int(w - col_indices[-1] - 1)

        # Валидация
        if (h - top - bottom) < h * min_ratio or (w - left - right) < w * min_ratio:
            return (0, 0, 0, 0)

        return (top, bottom, left, right)

    def _find_content_by_analysis(
        self,
        img: np.ndarray,
        ui_mask: np.ndarray,
        min_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Fallback: анализ яркости для поиска контента."""
        h, w = img.shape[:2]

        gray = np.mean(img[:, :, :3], axis=2) if len(img.shape) > 2 and img.shape[2] >= 3 else img
        row_brightness = np.mean(gray, axis=1)
        row_std = np.std(gray, axis=1)
        row_ui = np.mean(ui_mask, axis=1)

        # Определяем тип UI (тёмный/светлый)
        edge_sample = min(self.EDGE_SAMPLE_SIZE, h // 10)
        edge_brightness = np.mean([row_brightness[:edge_sample].mean(), row_brightness[-edge_sample:].mean()])
        is_dark_ui = edge_brightness < self.DARK_UI_THRESHOLD

        # Поиск верхней границы
        top = 0
        max_scan = min(h // 3, 400)
        for i in range(max_scan):
            is_ui = row_ui[i] > self.UI_DENSITY_THRESHOLD
            is_bg = row_std[i] < self.BRIGHTNESS_STD_THRESHOLD and (
                (is_dark_ui and row_brightness[i] < self.DARK_BRIGHTNESS_MAX) or
                (not is_dark_ui and row_brightness[i] > self.LIGHT_BRIGHTNESS_MIN)
            )
            if is_ui or is_bg:
                top = i + 1
            elif top > 0:
                break

        # Поиск нижней границы
        bottom = 0
        for i in range(max_scan):
            idx = h - 1 - i
            is_ui = row_ui[idx] > self.UI_DENSITY_THRESHOLD
            is_bg = row_std[idx] < self.BRIGHTNESS_STD_THRESHOLD and (
                (is_dark_ui and row_brightness[idx] < self.DARK_BRIGHTNESS_MAX) or
                (not is_dark_ui and row_brightness[idx] > self.LIGHT_BRIGHTNESS_MIN)
            )
            if is_ui or is_bg:
                bottom = i + 1
            elif bottom > 0:
                break

        # Валидация
        if (h - top - bottom) < h * min_ratio:
            return (0, 0, 0, 0)

        return (top, bottom, 0, 0)

    # ==================== Main Process ====================

    @torch.no_grad()
    def process(
        self,
        image: torch.Tensor,
        mode: str = "hybrid",
        yolo_confidence: float = 0.25,
        sod_threshold: float = 0.5,
        expand_mask_px: int = 5,
        crop_to_content: bool = True,
        min_content_ratio: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
        """
        Основной метод обработки. Поддерживает batch.

        Returns:
            cleaned_image, original_cropped, ui_mask, content_mask,
            crop_top, crop_bottom, crop_left, crop_right
        """

        batch_size = image.shape[0]
        cleaned_list = []
        original_list = []
        ui_mask_list = []
        content_mask_list = []
        final_crop = None

        use_yolo = mode in ["hybrid", "yolo_only", "full_inpaint"]
        use_sod = mode in ["hybrid", "sod_only", "full_inpaint"]
        use_inpaint = mode == "full_inpaint"

        for b in range(batch_size):
            img = image[b].cpu().numpy()
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1

            # Нормализуем к 3 каналам
            if c == 1:
                img = np.stack([img.squeeze()] * 3, axis=-1)
            elif c == 4:
                img = img[:, :, :3]

            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            # 1. Детекция UI
            ui_mask = np.zeros((h, w), dtype=np.float32)
            if use_yolo:
                _, ui_mask = self._detect_ui_yolo(img_uint8, yolo_confidence)

            # 2. Детекция контента
            content_mask = np.ones((h, w), dtype=np.float32)
            if use_sod:
                content_mask = self._detect_content_sod(img_uint8, sod_threshold)

            # 3. Инпеинтинг
            processed_img = img_uint8.copy()
            if use_inpaint and ui_mask.max() > 0:
                processed_img = self._inpaint_ui(img_uint8, ui_mask, expand_mask_px)
                logger.info(f"Inpainted {int(ui_mask.sum())} pixels")

            # 4. Определение границ обрезки
            crop_top, crop_bottom, crop_left, crop_right = 0, 0, 0, 0
            if crop_to_content:
                if use_sod and content_mask.mean() < 0.95:
                    crop_top, crop_bottom, crop_left, crop_right = self._find_content_bounds(
                        h, w, ui_mask, content_mask, min_content_ratio
                    )
                else:
                    crop_top, crop_bottom, crop_left, crop_right = self._find_content_by_analysis(
                        img_uint8, ui_mask, min_content_ratio
                    )

            # Сохраняем crop от первого изображения
            if final_crop is None:
                final_crop = (crop_top, crop_bottom, crop_left, crop_right)

            # 5. Применяем обрезку
            y_end = h - crop_bottom if crop_bottom > 0 else h
            x_end = w - crop_right if crop_right > 0 else w

            cleaned_cropped = processed_img[crop_top:y_end, crop_left:x_end]
            original_cropped = img_uint8[crop_top:y_end, crop_left:x_end]
            ui_mask_cropped = ui_mask[crop_top:y_end, crop_left:x_end]
            content_mask_cropped = content_mask[crop_top:y_end, crop_left:x_end]

            # Конвертируем в тензоры
            cleaned_list.append(torch.from_numpy(cleaned_cropped.astype(np.float32) / 255.0))
            original_list.append(torch.from_numpy(original_cropped.astype(np.float32) / 255.0))
            ui_mask_list.append(torch.from_numpy(ui_mask_cropped))
            content_mask_list.append(torch.from_numpy(content_mask_cropped))

        # Собираем batch
        cleaned_tensor = torch.stack(cleaned_list, dim=0)
        original_tensor = torch.stack(original_list, dim=0)
        ui_mask_tensor = torch.stack(ui_mask_list, dim=0)
        content_mask_tensor = torch.stack(content_mask_list, dim=0)

        crop_top, crop_bottom, crop_left, crop_right = final_crop or (0, 0, 0, 0)

        logger.info(f"Result: {cleaned_tensor.shape}, crop: t={crop_top}, b={crop_bottom}, l={crop_left}, r={crop_right}")

        return (
            cleaned_tensor,
            original_tensor,
            ui_mask_tensor,
            content_mask_tensor,
            crop_top,
            crop_bottom,
            crop_left,
            crop_right
        )

    def unload_models(self):
        """Выгружает модели из памяти."""
        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None

        if self.sod_session is not None:
            del self.sod_session
            self.sod_session = None

        if self.lama_model is not None:
            del self.lama_model
            self.lama_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded, GPU memory cleared")

import torch
import numpy as np
from typing import Tuple, Optional, List
import os


class AutoBorderCrop:
    """
    Автоматически обрезает однотонные рамки любого цвета.
    """

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
                }),
                "min_border_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "min_content_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 10.0,
                    "max": 90.0,
                    "step": 5.0,
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

        img = image[0].cpu().numpy()
        h, w, c = img.shape
        img_uint8 = (img * 255).astype(np.uint8)

        crop_top = self._detect_border(img_uint8, 'top', sensitivity, min_border_size)
        crop_bottom = self._detect_border(img_uint8, 'bottom', sensitivity, min_border_size)
        crop_left = self._detect_border(img_uint8, 'left', sensitivity, min_border_size)
        crop_right = self._detect_border(img_uint8, 'right', sensitivity, min_border_size)

        remaining_h = h - crop_top - crop_bottom
        remaining_w = w - crop_left - crop_right
        min_h = int(h * min_content_percent / 100)
        min_w = int(w * min_content_percent / 100)

        if remaining_h < min_h or remaining_w < min_w or remaining_h <= 0 or remaining_w <= 0:
            return (image, 0, 0, 0, 0, False)

        was_cropped = crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0

        if not was_cropped:
            return (image, 0, 0, 0, 0, False)

        y_end = h - crop_bottom if crop_bottom > 0 else h
        x_end = w - crop_right if crop_right > 0 else w
        cropped = image[:, crop_top:y_end, crop_left:x_end, :]

        return (cropped, crop_top, crop_bottom, crop_left, crop_right, was_cropped)

    def _detect_border(self, img: np.ndarray, side: str, sensitivity: float, min_border_size: int) -> int:
        h, w = img.shape[:2]

        if side == 'top':
            get_line = lambda i: img[i, :, :3] if img.shape[2] >= 3 else img[i, :, :]
            max_scan = h // 2
        elif side == 'bottom':
            get_line = lambda i: img[h - 1 - i, :, :3] if img.shape[2] >= 3 else img[h - 1 - i, :, :]
            max_scan = h // 2
        elif side == 'left':
            get_line = lambda i: img[:, i, :3] if img.shape[2] >= 3 else img[:, i, :]
            max_scan = w // 2
        else:
            get_line = lambda i: img[:, w - 1 - i, :3] if img.shape[2] >= 3 else img[:, w - 1 - i, :]
            max_scan = w // 2

        first_line = get_line(0).astype(np.float32)
        reference_color = np.median(first_line, axis=0)
        first_line_std = np.std(first_line, axis=0).mean()

        if first_line_std > sensitivity * 2:
            return 0

        border_end = 0
        for i in range(max_scan):
            line = get_line(i).astype(np.float32)
            line_avg = np.mean(line, axis=0)
            color_diff = np.abs(line_avg - reference_color).mean()
            line_std = np.std(line, axis=0).mean()

            if color_diff < sensitivity and line_std < sensitivity * 2:
                border_end = i + 1
            else:
                break

        if border_end >= min_border_size:
            return border_end
        return 0


class SmartScreenshotCleaner:
    """
    Умная нода для очистки скриншотов от UI элементов.

    Возможности:
    1. Детекция UI элементов через YOLO (deki-yolo)
    2. Детекция главного контента через SOD (Salient Object Detection)
    3. Инпеинтинг UI элементов через LaMa (заливка цветом окружения)
    4. Умная обрезка до области контента

    Модели скачиваются автоматически при первом запуске.
    """

    MODEL_REPO_YOLO = "orasul/deki-yolo"
    MODEL_FILE_YOLO = "best.pt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["hybrid", "yolo_only", "sod_only", "full_inpaint"], {
                    "default": "hybrid",
                    "tooltip": "hybrid - YOLO+SOD, full_inpaint - с заливкой UI"
                }),
                "use_sod": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Использовать Salient Object Detection для поиска контента"
                }),
                "use_inpainting": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Заливать UI элементы цветом окружения (LaMa)"
                }),
                "yolo_confidence": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                }),
                "sod_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Порог для SOD маски (что считать контентом)"
                }),
                "expand_mask_px": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Расширить маску UI на N пикселей для лучшего инпеинтинга"
                }),
                "crop_to_content": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Обрезать до области контента"
                }),
                "min_content_ratio": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.1,
                    "max": 0.8,
                    "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cleaned_image", "original_cropped", "ui_mask", "content_mask", "crop_top", "crop_bottom", "crop_left", "crop_right")
    FUNCTION = "process"
    CATEGORY = "image/preprocessing"

    def __init__(self):
        self.yolo_model = None
        self.sod_model = None
        self.lama_model = None
        self._models_dir = self._get_models_dir()

    def _get_models_dir(self) -> str:
        """Получает директорию для моделей."""
        # Пробуем использовать папку ComfyUI/models
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "screenshot_cleaner")
        except ImportError:
            # Fallback на локальную папку
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        os.makedirs(models_dir, exist_ok=True)
        return models_dir

    def _ensure_dependencies(self):
        """Проверяет и устанавливает зависимости если нужно."""
        missing = []

        try:
            import ultralytics
        except ImportError:
            missing.append("ultralytics")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            missing.append("huggingface_hub")

        if missing:
            print(f"[SmartScreenshotCleaner] Installing missing dependencies: {missing}")
            import subprocess
            import sys
            for pkg in missing:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

    # ==================== YOLO ====================

    def _load_yolo(self):
        """Загружает YOLO модель для детекции UI. Скачивает автоматически если нет."""
        if self.yolo_model is not None:
            return self.yolo_model

        self._ensure_dependencies()

        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download

            print(f"[SmartScreenshotCleaner] Downloading YOLO model from {self.MODEL_REPO_YOLO}...")

            model_path = hf_hub_download(
                repo_id=self.MODEL_REPO_YOLO,
                filename=self.MODEL_FILE_YOLO,
                cache_dir=self._models_dir
            )

            self.yolo_model = YOLO(model_path)
            print(f"[SmartScreenshotCleaner] YOLO loaded: {model_path}")
            return self.yolo_model

        except Exception as e:
            print(f"[SmartScreenshotCleaner] YOLO load error: {e}")
            return None

    def _detect_ui_yolo(self, img_np: np.ndarray, conf: float) -> Tuple[List[dict], np.ndarray]:
        """Детектирует UI элементы через YOLO."""
        model = self._load_yolo()
        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        if model is None:
            return [], mask

        results = model.predict(img_np, conf=conf, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().astype(int)
                conf_score = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                class_names = ["View", "ImageView", "Text", "Line"]
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

        print(f"[SmartScreenshotCleaner] YOLO detected {len(detections)} UI elements")
        return detections, mask

    # ==================== SOD ====================

    def _load_sod(self):
        """Загружает SOD модель. Скачивает автоматически если нет."""
        if self.sod_model is not None:
            return self.sod_model

        # Вариант 1: rembg (рекомендуется)
        try:
            from rembg import new_session
            print("[SmartScreenshotCleaner] Loading SOD model (rembg/u2net)...")
            self.sod_model = ("rembg", new_session("u2net"))
            print("[SmartScreenshotCleaner] SOD loaded: rembg/u2net")
            return self.sod_model
        except ImportError:
            print("[SmartScreenshotCleaner] rembg not found, trying to install...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "rembg", "-q"])
                from rembg import new_session
                self.sod_model = ("rembg", new_session("u2net"))
                print("[SmartScreenshotCleaner] SOD loaded: rembg/u2net (auto-installed)")
                return self.sod_model
            except Exception as e:
                print(f"[SmartScreenshotCleaner] Failed to install rembg: {e}")

        # Вариант 2: transparent-background
        try:
            from transparent_background import Remover
            self.sod_model = ("transparent_bg", Remover())
            print("[SmartScreenshotCleaner] SOD loaded: transparent-background")
            return self.sod_model
        except ImportError:
            pass

        print("[SmartScreenshotCleaner] SOD not available. Install: pip install rembg")
        return None

    def _detect_content_sod(self, img_np: np.ndarray, threshold: float) -> np.ndarray:
        """Детектирует главный контент через SOD."""
        h, w = img_np.shape[:2]

        sod = self._load_sod()
        if sod is None:
            return np.ones((h, w), dtype=np.float32)

        try:
            from PIL import Image
            img_pil = Image.fromarray(img_np)

            if sod[0] == "rembg":
                from rembg import remove
                result = remove(img_pil, session=sod[1], only_mask=True)
                mask = np.array(result).astype(np.float32) / 255.0

            elif sod[0] == "transparent_bg":
                remover = sod[1]
                result = remover.process(img_pil, type='map')
                mask = np.array(result).astype(np.float32) / 255.0

            else:
                return np.ones((h, w), dtype=np.float32)

            mask_binary = (mask > threshold).astype(np.float32)
            print(f"[SmartScreenshotCleaner] SOD content coverage: {mask_binary.mean()*100:.1f}%")
            return mask_binary

        except Exception as e:
            print(f"[SmartScreenshotCleaner] SOD error: {e}")
            return np.ones((h, w), dtype=np.float32)

    # ==================== LaMa Inpainting ====================

    def _load_lama(self):
        """Загружает LaMa модель. Скачивает автоматически если нет."""
        if self.lama_model is not None:
            return self.lama_model

        try:
            from simple_lama_inpainting import SimpleLama
            print("[SmartScreenshotCleaner] Loading LaMa model...")
            self.lama_model = SimpleLama()
            print("[SmartScreenshotCleaner] LaMa loaded: simple-lama-inpainting")
            return self.lama_model
        except ImportError:
            print("[SmartScreenshotCleaner] simple-lama-inpainting not found, trying to install...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "simple-lama-inpainting", "-q"])
                from simple_lama_inpainting import SimpleLama
                self.lama_model = SimpleLama()
                print("[SmartScreenshotCleaner] LaMa loaded (auto-installed)")
                return self.lama_model
            except Exception as e:
                print(f"[SmartScreenshotCleaner] Failed to install LaMa: {e}")

        # Fallback: OpenCV
        try:
            import cv2
            self.lama_model = ("opencv", None)
            print("[SmartScreenshotCleaner] Using OpenCV inpainting fallback")
            return self.lama_model
        except Exception:
            pass

        return None

    def _inpaint_ui(self, img_np: np.ndarray, mask: np.ndarray, expand_px: int) -> np.ndarray:
        """Закрашивает UI элементы через LaMa или fallback."""
        import cv2
        from PIL import Image

        h, w = img_np.shape[:2]

        # Расширяем маску
        if expand_px > 0:
            kernel = np.ones((expand_px * 2 + 1, expand_px * 2 + 1), np.uint8)
            mask_expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        else:
            mask_expanded = mask.astype(np.uint8)

        if mask_expanded.max() == 0:
            return img_np

        lama = self._load_lama()

        if lama is not None and lama != ("opencv", None):
            try:
                img_pil = Image.fromarray(img_np)
                mask_pil = Image.fromarray((mask_expanded * 255).astype(np.uint8))
                result = lama(img_pil, mask_pil)
                return np.array(result)
            except Exception as e:
                print(f"[SmartScreenshotCleaner] LaMa inpaint error: {e}")

        # Fallback: OpenCV TELEA
        try:
            mask_uint8 = (mask_expanded * 255).astype(np.uint8)
            result = cv2.inpaint(img_np, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            print("[SmartScreenshotCleaner] Used OpenCV TELEA inpainting")
            return result
        except Exception as e:
            print(f"[SmartScreenshotCleaner] OpenCV inpaint error: {e}")
            return img_np

    # ==================== Content Region Detection ====================

    def _find_content_bounds(
        self,
        h: int, w: int,
        ui_mask: np.ndarray,
        content_mask: np.ndarray,
        min_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Находит границы контента."""
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

        remaining_h = h - top - bottom
        remaining_w = w - left - right

        if remaining_h < h * min_ratio or remaining_w < w * min_ratio:
            return (0, 0, 0, 0)

        return (top, bottom, left, right)

    def _find_content_by_analysis(
        self,
        img: np.ndarray,
        ui_mask: np.ndarray,
        min_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Fallback: анализ яркости + UI маска."""
        h, w = img.shape[:2]

        gray = np.mean(img[:, :, :3], axis=2) if img.shape[2] >= 3 else img[:, :, 0]
        row_brightness = np.mean(gray, axis=1)
        row_std = np.std(gray, axis=1)
        row_ui = np.mean(ui_mask, axis=1)

        edge_brightness = np.mean([row_brightness[:20].mean(), row_brightness[-20:].mean()])
        is_dark_ui = edge_brightness < 60

        top = 0
        for i in range(min(h // 3, 400)):
            is_ui = row_ui[i] > 0.2
            is_bg = row_std[i] < 20 and (
                (is_dark_ui and row_brightness[i] < 70) or
                (not is_dark_ui and row_brightness[i] > 200)
            )
            if is_ui or is_bg:
                top = i + 1
            elif top > 0:
                break

        bottom = 0
        for i in range(min(h // 3, 400)):
            idx = h - 1 - i
            is_ui = row_ui[idx] > 0.2
            is_bg = row_std[idx] < 20 and (
                (is_dark_ui and row_brightness[idx] < 70) or
                (not is_dark_ui and row_brightness[idx] > 200)
            )
            if is_ui or is_bg:
                bottom = i + 1
            elif bottom > 0:
                break

        if (h - top - bottom) < h * min_ratio:
            return (0, 0, 0, 0)

        return (top, bottom, 0, 0)

    # ==================== Main Process ====================

    def process(
        self,
        image: torch.Tensor,
        mode: str = "hybrid",
        use_sod: bool = True,
        use_inpainting: bool = False,
        yolo_confidence: float = 0.25,
        sod_threshold: float = 0.5,
        expand_mask_px: int = 5,
        crop_to_content: bool = True,
        min_content_ratio: float = 0.2,
    ):
        """Основной метод обработки."""

        img = image[0].cpu().numpy()
        h, w, c = img.shape
        img_uint8 = (img * 255).astype(np.uint8)

        # 1. Детекция UI через YOLO
        ui_detections, ui_mask = [], np.zeros((h, w), dtype=np.float32)

        if mode in ["hybrid", "yolo_only", "full_inpaint"]:
            ui_detections, ui_mask = self._detect_ui_yolo(img_uint8, yolo_confidence)

        # 2. Детекция контента через SOD
        content_mask = np.ones((h, w), dtype=np.float32)

        if use_sod and mode in ["hybrid", "sod_only", "full_inpaint"]:
            content_mask = self._detect_content_sod(img_uint8, sod_threshold)

        # 3. Инпеинтинг UI элементов
        processed_img = img_uint8.copy()

        if use_inpainting and mode == "full_inpaint" and ui_mask.max() > 0:
            processed_img = self._inpaint_ui(img_uint8, ui_mask, expand_mask_px)
            print(f"[SmartScreenshotCleaner] Inpainted {int(ui_mask.sum())} pixels")

        # 4. Определение границ для обрезки
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

        # 5. Применяем обрезку
        y_start = crop_top
        y_end = h - crop_bottom if crop_bottom > 0 else h
        x_start = crop_left
        x_end = w - crop_right if crop_right > 0 else w

        cleaned_cropped = processed_img[y_start:y_end, x_start:x_end]
        original_cropped = img_uint8[y_start:y_end, x_start:x_end]
        ui_mask_cropped = ui_mask[y_start:y_end, x_start:x_end]
        content_mask_cropped = content_mask[y_start:y_end, x_start:x_end]

        cleaned_tensor = torch.from_numpy(cleaned_cropped.astype(np.float32) / 255.0).unsqueeze(0)
        original_tensor = torch.from_numpy(original_cropped.astype(np.float32) / 255.0).unsqueeze(0)
        ui_mask_tensor = torch.from_numpy(ui_mask_cropped).unsqueeze(0)
        content_mask_tensor = torch.from_numpy(content_mask_cropped).unsqueeze(0)

        print(f"[SmartScreenshotCleaner] Result: {cleaned_cropped.shape[1]}x{cleaned_cropped.shape[0]}, "
              f"cropped: top={crop_top}, bottom={crop_bottom}, left={crop_left}, right={crop_right}")

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

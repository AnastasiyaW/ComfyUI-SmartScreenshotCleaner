import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import os
import logging
from typing import Optional, Tuple, List, Dict
from pathlib import Path

# LaMa import with availability check
try:
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    HAS_LAMA = True
except ImportError:
    HAS_LAMA = False
    SimpleLama = None
    Image = None

logger = logging.getLogger("HappyIn")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_surrounding_color_gpu(img: torch.Tensor, x1: int, y1: int, x2: int, y2: int, margin: int = 10) -> torch.Tensor:
    """Получает РЕАЛЬНЫЙ цвет пикселей вокруг области на GPU (без нормализации к чёрному/белому)."""
    h, w = img.shape[:2]
    pixels = []

    if y1 > 0:
        pixels.append(img[max(0, y1-margin):y1, x1:x2, :3].reshape(-1, 3))
    if y2 < h:
        pixels.append(img[y2:min(h, y2+margin), x1:x2, :3].reshape(-1, 3))
    if x1 > 0:
        pixels.append(img[y1:y2, max(0, x1-margin):x1, :3].reshape(-1, 3))
    if x2 < w:
        pixels.append(img[y1:y2, x2:min(w, x2+margin), :3].reshape(-1, 3))

    if not pixels:
        return torch.full((3,), 128, device=img.device, dtype=img.dtype)

    all_pixels = torch.cat(pixels, dim=0).float()
    # Возвращаем реальный медианный цвет БЕЗ нормализации
    median_color = all_pixels.median(dim=0).values

    return median_color


def _morphology_open_close_gpu(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Морфологические операции open + close на GPU через max/min pooling."""
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
    padding = kernel_size // 2

    eroded = -F.max_pool2d(-mask_4d, kernel_size, stride=1, padding=padding)
    opened = F.max_pool2d(eroded, kernel_size, stride=1, padding=padding)
    dilated = F.max_pool2d(opened, kernel_size, stride=1, padding=padding)
    closed = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=padding)

    return closed.squeeze(0).squeeze(0)


# Общий кэш для YOLO моделей
_yolo_models_cache = {}


def _load_yolo_model(model_name: str, device: torch.device):
    """Загружает YOLO модель с кэшированием. Автоматически скачивает если нет."""
    cache_key = f"{model_name}_{device}"

    if cache_key in _yolo_models_cache:
        return _yolo_models_cache[cache_key]

    try:
        from ultralytics import YOLO

        if model_name == "yolo_person":
            # YOLOv11n — быстрый и точный для детекции людей
            # Автоматически скачается при первом использовании
            model = YOLO("yolo11n.pt")
            logger.info("Using YOLOv11n for person detection (auto-download if needed)")
        elif model_name == "omniparser":
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                "microsoft/OmniParser-v2.0",
                "icon_detect/model.pt",
                cache_dir=os.path.join(os.path.dirname(__file__), "..", "models")
            )
            model = YOLO(path)
        else:
            # Fallback на указанную модель
            model = YOLO(f"{model_name}.pt")

        if device.type in ("cuda", "mps"):
            model.to(device)

        _yolo_models_cache[cache_key] = model
        logger.info(f"YOLO {model_name} loaded on {device}")
        return model

    except Exception as e:
        logger.error(f"Failed to load YOLO {model_name}: {e}")
        return None


class AutoBorderCrop:
    """Обрезает однотонные чёрные/белые/серые рамки с краёв изображения.

    ПРОСТАЯ логика:
    1. Сканируем от края к центру по линиям
    2. Серый/чёрный/белый (R≈G≈B, низкая динамика) = рамка → РЕЖЕМ
    3. Любой цвет ИЛИ высокая динамика = картинка → СТОП
    4. Safe margins от человека ТОЛЬКО если вокруг него рамка (не картинка)
    """

    # Safe margins — ВСЕГДА оставляем отступ от человека
    SAFE_MARGIN = 200  # 200px в каждую сторону от bbox человека

    # Пороги для определения рамки
    GRAY_MAX_SATURATION = 30.0    # max(R,G,B) - min(R,G,B) < 30 = серый/ч/б
    GRAY_MAX_STD = 25.0           # STD < 25 = однотонная область (рамка, включая артефакты сжатия)

    # Пороги для определения картинки
    PICTURE_MIN_SATURATION = 35.0  # Насыщенность > 35 = есть цвет
    PICTURE_MIN_STD = 40.0         # STD > 40 = есть динамика (настоящая картинка имеет STD >> 40)

    MIN_BORDER_TO_CUT = 3          # Минимум 3px для обрезки
    MIN_CONTENT_RATIO = 0.3        # Минимум 30% контента после обрезки
    SCAN_LINE_WIDTH = 5            # Ширина сканируемой линии в пикселях

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "min_border_size": ("INT", {"default": 5, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "debug_mask", "debug_bbox")
    FUNCTION = "crop_borders"
    CATEGORY = "image/preprocessing"

    @torch.no_grad()
    def crop_borders(self, image: torch.Tensor, sensitivity: float = 15.0, min_border_size: int = 5):
        """Обрезает однотонные рамки с краёв изображения."""
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image)}")
        if len(image.shape) != 4:
            raise ValueError(f"Expected image shape [B, H, W, C], got {image.shape}")
        if image.shape[1] < 10 or image.shape[2] < 10:
            logger.warning(f"Image too small: {image.shape}, returning original")
            h, w = image.shape[1], image.shape[2]
            empty_mask = torch.zeros(image.shape[0], h, w)
            return (image, empty_mask, image.clone())

        device = get_device()
        single_image = image[0]
        result, debug_mask, debug_bbox = self._process_single_image_gpu(single_image, sensitivity, min_border_size, device)
        return (result.unsqueeze(0), debug_mask.unsqueeze(0), debug_bbox.unsqueeze(0))

    def _detect_person_bbox_yolo(self, img_gpu: torch.Tensor, device: torch.device) -> Optional[Tuple[int, int, int, int]]:
        """Детектирует человека через YOLO v11 и возвращает bbox."""
        try:
            model = _load_yolo_model("yolo_person", device)
            if model is None:
                return None

            img_np = img_gpu.cpu().numpy().astype(np.uint8)
            results = model.predict(img_np, conf=0.3, verbose=False, device=device, imgsz=640, classes=[0])

            if not results or len(results) == 0:
                return None

            best_box = None
            best_area = 0

            for result in results:
                if result.boxes is None:
                    continue
                for i in range(len(result.boxes)):
                    box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_box = (y1, y2, x1, x2)  # (top, bottom, left, right)

            if best_box:
                logger.info(f"YOLO detected person bbox: {best_box}")
            return best_box

        except Exception as e:
            logger.warning(f"YOLO person detection failed: {e}")
            return None

    def _analyze_line(self, line_pixels: torch.Tensor) -> Tuple[float, float, float]:
        """Анализирует линию пикселей.

        Returns:
            (saturation, std, uniform_ratio): насыщенность, динамика, доля однотонных пикселей
        """
        if line_pixels.numel() == 0:
            return 0.0, 0.0, 0.0

        pixels = line_pixels.reshape(-1, 3).float()

        # Насыщенность: max(R,G,B) - min(R,G,B) для каждого пикселя
        sat_per_pixel = pixels.max(dim=1).values - pixels.min(dim=1).values
        saturation = sat_per_pixel.mean().item()

        # Динамика (STD) — разброс значений
        std = pixels.std().item()

        # Доля однотонных пикселей (серые, R≈G≈B)
        # Пиксель однотонный если его насыщенность < 30
        uniform_pixels = (sat_per_pixel < 30.0).float()
        uniform_ratio = uniform_pixels.mean().item()

        return saturation, std, uniform_ratio

    def _is_border_line(self, saturation: float, std: float, uniform_ratio: float) -> bool:
        """Проверяет, является ли линия частью рамки.

        КЛЮЧЕВОЙ КРИТЕРИЙ: Рамка = СЕРЫЙ/ЧЁРНЫЙ/БЕЛЫЙ (R≈G≈B)
        Если есть цвет (насыщенность > 20) — это НЕ рамка, даже если однотонно!

        Рамка если:
        - Низкая насыщенность (серый, R≈G≈B) + низкая динамика
        - >95% пикселей серые (uniform_ratio высокий) + низкая насыщенность
        """
        # ВАЖНО: Если есть цвет — это НЕ рамка (даже однотонный зелёный фон = фото)
        if saturation > 25.0:
            return False

        # Серый + однотонный = рамка
        is_gray = saturation < self.GRAY_MAX_SATURATION  # < 30
        is_uniform = std < self.GRAY_MAX_STD  # < 25

        if is_gray and is_uniform:
            return True

        # >95% серых пикселей + низкая насыщенность
        if uniform_ratio >= 0.95 and saturation < 20.0:
            return True

        return False

    def _is_picture_line(self, saturation: float, std: float, uniform_ratio: float) -> bool:
        """Проверяет, является ли линия частью картинки.

        КЛЮЧЕВОЙ КРИТЕРИЙ: Если есть ЦВЕТ (насыщенность > 25) — это картинка!
        Даже однотонный зелёный/синий фон = часть фото, НЕ резать.

        Картинка если:
        - Есть цвет (насыщенность > 25) — даже если однотонно
        - ИЛИ высокая динамика (STD > 40)
        - ИЛИ много цветных пикселей (<60% серых)
        """
        # ГЛАВНОЕ: Есть цвет — это картинка! (зелёный фон, синее небо и т.д.)
        if saturation > 25.0:
            return True

        # Высокая динамика — картинка
        if std >= self.PICTURE_MIN_STD:
            return True

        # Много цветных пикселей
        if uniform_ratio < 0.60:
            return True

        return False

    def _scan_from_edge(self, img: torch.Tensor, side: str) -> Tuple[int, bool]:
        """Сканирует от края к центру, ищет границу рамки.

        Логика: идём от края, пока >90% пикселей однотонные — это рамка.
        Как только <70% однотонных — это картинка, СТОП.

        Returns:
            (border_size, found_picture):
            - border_size: сколько пикселей рамки можно обрезать
            - found_picture: нашли ли явную границу картинки
        """
        h, w = img.shape[:2]
        line_w = self.SCAN_LINE_WIDTH

        # Сканируем ВСЮ линию — uniform_ratio покажет долю рамки
        if side == 'top':
            max_scan = h // 2
            get_line = lambda pos: img[pos:pos+line_w, :, :3]
        elif side == 'bottom':
            max_scan = h // 2
            get_line = lambda pos: img[h-pos-line_w:h-pos, :, :3]
        elif side == 'left':
            max_scan = w // 2
            get_line = lambda pos: img[:, pos:pos+line_w, :3]
        else:  # right
            max_scan = w // 2
            get_line = lambda pos: img[:, w-pos-line_w:w-pos, :3]

        border_size = 0
        found_picture = False

        for pos in range(0, max_scan):
            line = get_line(pos)
            if line.numel() == 0:
                break

            saturation, std, uniform_ratio = self._analyze_line(line)

            # Это картинка? (<70% однотонных пикселей)
            if self._is_picture_line(saturation, std, uniform_ratio):
                logger.info(f"{side}: PICTURE at {pos}px (sat={saturation:.1f}, std={std:.1f}, uniform={uniform_ratio:.0%})")
                found_picture = True
                break

            # Это рамка? (>90% однотонных)
            if self._is_border_line(saturation, std, uniform_ratio):
                border_size = pos + 1
            # Промежуточная зона (70-90%) — продолжаем, но не увеличиваем border

        return (border_size if border_size >= self.MIN_BORDER_TO_CUT else 0, found_picture)

    def _process_single_image_gpu(self, image: torch.Tensor, sensitivity: float, min_border_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Обрабатывает одно изображение.

        ПРОСТОЙ алгоритм:
        1. Детектируем человека через YOLO
        2. Для каждой стороны сканируем от края
        3. Серый/однотонный → режем, цвет/динамика → стоп
        4. Safe margins от человека ТОЛЬКО если вокруг него рамка
        """
        img_gpu = (image.to(device) * 255.0).float()
        h, w = img_gpu.shape[:2]

        debug_mask = torch.zeros(h, w, device=device)
        person_bbox = self._detect_person_bbox_yolo(img_gpu, device)
        debug_bbox_img = self._create_debug_bbox_image(img_gpu, person_bbox)

        crop = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

        for side in ['top', 'bottom', 'left', 'right']:
            # Сканируем от края
            border, found_picture = self._scan_from_edge(img_gpu, side)

            if border < self.MIN_BORDER_TO_CUT:
                continue

            # Определяем финальную обрезку
            if found_picture:
                # Нашли границу картинки — режем точно до неё, оставляем ВСЕ цветные пиксели
                final_border = border
                logger.info(f"{side}: cut to PICTURE boundary: {final_border}px")
            else:
                # НЕ нашли картинку — всё серое, применяем safe margin от человека
                final_border = border

                if person_bbox:
                    py1, py2, px1, px2 = person_bbox
                    if side == 'top':
                        safe_limit = max(0, py1 - self.SAFE_MARGIN)
                    elif side == 'bottom':
                        safe_limit = max(0, h - py2 - self.SAFE_MARGIN)
                    elif side == 'left':
                        safe_limit = max(0, px1 - self.SAFE_MARGIN)
                    else:  # right
                        safe_limit = max(0, w - px2 - self.SAFE_MARGIN)

                    final_border = min(final_border, safe_limit)
                    logger.info(f"{side}: no picture found, safe margin: {final_border}px (limit {safe_limit}px)")
                else:
                    logger.info(f"{side}: no picture, no person, cut border: {final_border}px")

            crop[side] = final_border

            # Debug маска
            if final_border > 0:
                if side == 'top':
                    debug_mask[:final_border, :] = 1.0
                elif side == 'bottom':
                    debug_mask[h-final_border:, :] = 1.0
                elif side == 'left':
                    debug_mask[:, :final_border] = 1.0
                else:
                    debug_mask[:, w-final_border:] = 1.0

        logger.info(f"Final crop: top={crop['top']}, bottom={crop['bottom']}, left={crop['left']}, right={crop['right']}")

        # Применяем обрезку
        new_h = h - crop['top'] - crop['bottom']
        new_w = w - crop['left'] - crop['right']

        if new_h < h * self.MIN_CONTENT_RATIO or new_w < w * self.MIN_CONTENT_RATIO:
            logger.warning(f"Crop too aggressive: {new_h}x{new_w}, returning original")
            return ((img_gpu / 255.0).clamp(0, 1).cpu(), debug_mask.cpu(), debug_bbox_img.cpu())

        y1 = crop['top']
        y2 = h - crop['bottom'] if crop['bottom'] > 0 else h
        x1 = crop['left']
        x2 = w - crop['right'] if crop['right'] > 0 else w

        if y1 >= y2 or x1 >= x2:
            logger.warning(f"Invalid crop bounds, returning original")
            return ((img_gpu / 255.0).clamp(0, 1).cpu(), debug_mask.cpu(), debug_bbox_img.cpu())

        result = (img_gpu[y1:y2, x1:x2] / 255.0).clamp(0, 1).cpu()
        return (result, debug_mask.cpu(), debug_bbox_img.cpu())

    def _create_debug_bbox_image(self, img_gpu: torch.Tensor, person_bbox: Optional[Tuple]) -> torch.Tensor:
        """Создаёт debug изображение с bbox."""
        debug_img = img_gpu.clone()
        h, w = debug_img.shape[:2]
        line_width = max(2, min(h, w) // 200)

        def draw_rect(img, y1, y2, x1, x2, color, width=2):
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            img[y1:y1+width, x1:x2, :3] = color
            img[y2-width:y2, x1:x2, :3] = color
            img[y1:y2, x1:x1+width, :3] = color
            img[y1:y2, x2-width:x2, :3] = color

        if person_bbox:
            py1, py2, px1, px2 = person_bbox
            draw_rect(debug_img, py1, py2, px1, px2,
                     torch.tensor([0, 255, 0], device=img_gpu.device, dtype=img_gpu.dtype),
                     line_width)

        return (debug_img / 255.0).clamp(0, 1)


class SmartScreenshotCleaner:
    """
    Находит UI элементы через YOLO модели и удаляет их через LaMa инпейнтинг.
    Использует OmniParser для детекции UI элементов.
    """

    _models_cache = {}  # Кэш для всех моделей
    _model_device = None

    # Константы класса
    DEFAULT_CONFIDENCE = 0.15     # Порог уверенности для UI элементов
    MAX_BOX_RATIO = 0.05          # Максимальный размер UI элемента (5% картинки) — иконки маленькие!
    MAX_BOX_SIDE = 150            # Максимальная сторона бокса в пикселях
    SURROUNDING_MARGIN = 10
    BOX_EXPAND_PIXELS = 5         # Расширение маски вокруг UI элементов (уменьшено)
    YOLO_IMGSZ = 1280             # Разрешение для YOLO
    IOU_THRESHOLD = 0.5           # Порог IoU для дедупликации боксов

    # Зоны UI (доля изображения от края)
    UI_EDGE_ZONE = 0.20           # 20% от края считается UI зоной (уменьшено)
    UI_CENTER_SAFE_ZONE = 0.5     # Центральные 50% защищены от детекции (увеличено)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.9, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "ui_mask")
    FUNCTION = "process"
    CATEGORY = "image/preprocessing"

    def __init__(self):
        self.device = get_device()

    def _get_models_dir(self) -> str:
        try:
            import folder_paths
            d = os.path.join(folder_paths.models_dir, "screenshot_cleaner")
        except ImportError:
            d = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def unload_models(cls):
        """Освобождает память GPU от всех моделей."""
        for name, model in cls._models_cache.items():
            del model
            logger.info(f"Unloaded model: {name}")
        cls._models_cache = {}
        cls._model_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_name: str, repo_id: str, filename: str):
        """Загружает модель из HuggingFace Hub с кэшированием."""
        current_device = self.device
        cache_key = f"{model_name}_{current_device}"

        if cache_key in SmartScreenshotCleaner._models_cache:
            return SmartScreenshotCleaner._models_cache[cache_key]

        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id,
                filename,
                cache_dir=self._get_models_dir()
            )

            model = YOLO(path)
            if current_device.type in ("cuda", "mps"):
                model.to(current_device)

            SmartScreenshotCleaner._models_cache[cache_key] = model
            SmartScreenshotCleaner._model_device = current_device
            logger.info(f"Loaded {model_name} on {current_device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None

    def _load_all_models(self) -> List:
        """Загружает все модели для детекции UI элементов.

        Модели:
        1. OmniParser icon_detect — UI элементы десктопа (иконки, кнопки, меню)
        2. OmniParser icon_caption — текст и подписи в UI
        3. YOLOv11n — быстрая универсальная модель
        """
        models = []

        # 1. OmniParser icon_detect — основная модель для UI элементов
        omni_icon = self._load_model(
            "omniparser_icon",
            "microsoft/OmniParser-v2.0",
            "icon_detect/model.pt"
        )
        if omni_icon:
            models.append(("omniparser_icon", omni_icon))

        # 2. OmniParser icon_caption — детекция текста в UI
        omni_caption = self._load_model(
            "omniparser_caption",
            "microsoft/OmniParser-v2.0",
            "icon_caption/model.pt"
        )
        if omni_caption:
            models.append(("omniparser_caption", omni_caption))

        # 3. YOLOv11n — быстрая универсальная модель (автоскачивание)
        try:
            from ultralytics import YOLO
            cache_key = f"yolo11n_{self.device}"
            if cache_key not in SmartScreenshotCleaner._models_cache:
                yolo11 = YOLO("yolo11n.pt")  # Автоскачивание
                if self.device.type in ("cuda", "mps"):
                    yolo11.to(self.device)
                SmartScreenshotCleaner._models_cache[cache_key] = yolo11
                logger.info(f"Loaded YOLOv11n on {self.device}")
            models.append(("yolo11n", SmartScreenshotCleaner._models_cache[cache_key]))
        except Exception as e:
            logger.warning(f"YOLOv11n not loaded: {e}")

        logger.info(f"Loaded {len(models)} detection models")
        return models

    def _deduplicate_boxes(self, boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Удаляет дублирующиеся боксы по IoU (Intersection over Union)."""
        if len(boxes) <= 1:
            return boxes

        # Сортируем по площади (большие сначала)
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        keep = []

        for box in boxes:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)

            should_keep = True
            for kept_box in keep:
                kx1, ky1, kx2, ky2 = kept_box

                # Вычисляем пересечение
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)

                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    kept_area = (kx2 - kx1) * (ky2 - ky1)
                    union = box_area + kept_area - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > iou_threshold:
                        should_keep = False
                        break

            if should_keep:
                keep.append(box)

        return keep

    def _is_ui_zone(self, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, model_name: str) -> bool:
        """Проверяет, находится ли бокс в UI зоне (края изображения).

        Для моделей детекции текста (icon_caption) — строгая фильтрация:
        текст принимается только если он находится на краях изображения.

        Для icon_detect и YOLO — принимаем всё (они уже обучены на UI).
        """
        # icon_detect и YOLO обучены на UI элементах — принимаем всё
        if model_name != "omniparser_caption":
            return True

        # Для icon_caption — фильтруем текст в центре изображения
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        # Границы центральной "безопасной" зоны
        safe_left = img_w * self.UI_CENTER_SAFE_ZONE
        safe_right = img_w * (1 - self.UI_CENTER_SAFE_ZONE)
        safe_top = img_h * self.UI_CENTER_SAFE_ZONE
        safe_bottom = img_h * (1 - self.UI_CENTER_SAFE_ZONE)

        # Если центр бокса в безопасной зоне — это скорее всего контент, не UI
        if safe_left < box_center_x < safe_right and safe_top < box_center_y < safe_bottom:
            # Дополнительная проверка: маленький текст в центре может быть watermark
            box_area = (x2 - x1) * (y2 - y1)
            img_area = img_w * img_h
            if box_area / img_area < 0.01:  # Очень маленький текст (< 1%) — может быть watermark
                return True
            logger.debug(f"Skipping text in center: {x1},{y1}-{x2},{y2}")
            return False

        return True

    def _is_in_ui_panel(self, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> bool:
        """Проверяет, находится ли бокс в типичной UI панели (справа, снизу, сверху)."""
        # Правая панель (Instagram comments обычно справа)
        if x1 > img_w * 0.5:
            return True

        # Верхняя панель (status bar, навигация)
        if y2 < img_h * self.UI_EDGE_ZONE:
            return True

        # Нижняя панель (комментарии, лайки, навигация)
        if y1 > img_h * (1 - self.UI_EDGE_ZONE):
            return True

        return False

    @torch.no_grad()
    def process(self, image: torch.Tensor, confidence: float = 0.1):
        """
        Обрабатывает изображения: находит UI элементы и удаляет их через LaMa.

        Использует несколько моделей:
        1. OmniParser icon_detect — UI элементы (иконки, кнопки)
        2. YOLOv8n — общие объекты (текст, мелкие элементы)
        """
        if image.shape[0] == 0:
            return (image, torch.zeros(0, 1, 1, device=image.device))

        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]

        models = self._load_all_models()
        if not models:
            logger.warning("No models loaded, returning original image")
            empty_mask = torch.zeros(batch_size, h, w, device=image.device)
            return (image, empty_mask)

        logger.info(f"Using {len(models)} models: {[m[0] for m in models]}")

        all_results = []
        all_masks = []

        for batch_idx in range(batch_size):
            single_img = image[batch_idx]
            result_img, result_mask = self._process_single_image(single_img, models, confidence)
            all_results.append(result_img)
            all_masks.append(result_mask)

        out_images = torch.stack(all_results)
        out_masks = torch.stack(all_masks)

        return (out_images, out_masks)

    def _process_single_image(self, img: torch.Tensor, models: List, confidence: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Обрабатывает одно изображение на GPU.
        Использует несколько YOLO моделей для детекции UI элементов и LaMa для инпейнтинга."""
        device = self.device
        h, w = img.shape[:2]

        img_gpu = img.to(device)
        img_uint8_gpu = (img_gpu.clamp(0, 1) * 255).to(torch.uint8)

        img_np = img_uint8_gpu.cpu().numpy()

        all_boxes: List[Tuple[int, int, int, int]] = []

        # Прогоняем через все модели
        for model_name, model in models:
            if model is None:
                continue

            try:
                # Для YOLOv11n фильтруем только UI-подобные классы
                if model_name == "yolo11n":
                    # COCO классы UI-подобных объектов:
                    # tv(62), laptop(63), mouse(64), remote(65), keyboard(66), cell phone(67), book(73), clock(74)
                    results = model.predict(
                        img_np, conf=confidence * 0.5, verbose=False, device=device,
                        imgsz=self.YOLO_IMGSZ, classes=[62, 63, 64, 65, 66, 67, 73, 74]
                    )
                else:
                    # OmniParser — все классы с низким порогом
                    results = model.predict(
                        img_np, conf=confidence, verbose=False, device=device,
                        imgsz=self.YOLO_IMGSZ
                    )

                model_boxes = 0
                skipped_center = 0
                for result in results:
                    if result.boxes is None:
                        continue
                    for i in range(len(result.boxes)):
                        box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            # Фильтруем текст в центре (только для icon_caption)
                            if self._is_ui_zone(x1, y1, x2, y2, w, h, model_name):
                                all_boxes.append((x1, y1, x2, y2))
                                model_boxes += 1
                            else:
                                skipped_center += 1

                if skipped_center > 0:
                    logger.info(f"{model_name} detected: {model_boxes} UI elements (skipped {skipped_center} in center)")
                else:
                    logger.info(f"{model_name} detected: {model_boxes} UI elements")

            except Exception as e:
                logger.warning(f"{model_name} failed: {e}")

        if not all_boxes:
            logger.info("No UI elements detected, returning original")
            return img, torch.zeros(h, w, device=device)

        # Дедупликация боксов (удаляем сильно перекрывающиеся)
        all_boxes = self._deduplicate_boxes(all_boxes, iou_threshold=0.5)
        logger.info(f"Total UI elements after dedup: {len(all_boxes)}")

        # Создаём маску для всех UI элементов
        ui_mask = torch.zeros(h, w, device=device, dtype=torch.float32)
        img_area = h * w

        for x1, y1, x2, y2 in all_boxes:
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            # Пропускаем слишком большие боксы (это не иконки!)
            if box_area > img_area * self.MAX_BOX_RATIO:
                logger.info(f"Skipping large box {x1},{y1}-{x2},{y2}: {box_area/img_area*100:.1f}% of image")
                continue

            # Пропускаем боксы с большой стороной (иконки маленькие)
            if box_w > self.MAX_BOX_SIDE or box_h > self.MAX_BOX_SIDE:
                logger.info(f"Skipping box with large side {x1},{y1}-{x2},{y2}: {box_w}x{box_h}px")
                continue

            # Расширяем маску на несколько пикселей для лучшего инпейнтинга
            expand = self.BOX_EXPAND_PIXELS
            y1_exp = max(0, y1 - expand)
            y2_exp = min(h, y2 + expand)
            x1_exp = max(0, x1 - expand)
            x2_exp = min(w, x2 + expand)
            ui_mask[y1_exp:y2_exp, x1_exp:x2_exp] = 1.0
            logger.info(f"Added UI element to mask: {x1},{y1}-{x2},{y2}")

        if ui_mask.sum() == 0:
            logger.info("No valid UI elements to remove")
            return img, ui_mask.cpu()

        # Используем LaMa для инпейнтинга всех UI элементов за один проход
        out_img = self._apply_lama_inpainting(img_uint8_gpu, ui_mask)

        return out_img.cpu(), ui_mask.cpu()

    def _apply_lama_inpainting(self, img_gpu: torch.Tensor, mask_gpu: torch.Tensor) -> torch.Tensor:
        """Применяет LaMa инпейнтинг для удаления UI элементов."""
        if not HAS_LAMA:
            logger.warning("LaMa not available, falling back to color fill")
            return self._fallback_color_fill(img_gpu, mask_gpu)

        try:
            # Конвертируем в numpy для LaMa
            img_np = img_gpu.cpu().numpy().astype(np.uint8)
            mask_np = (mask_gpu.cpu().numpy() * 255).astype(np.uint8)

            logger.info(f"Applying LaMa inpainting, mask pixels: {(mask_np > 0).sum()}")

            lama = SimpleLama()
            result = lama(Image.fromarray(img_np), Image.fromarray(mask_np))

            # Конвертируем обратно в tensor [0, 1]
            result_np = np.array(result).astype(np.float32) / 255.0
            return torch.from_numpy(result_np)

        except Exception as e:
            logger.warning(f"LaMa failed: {e}, falling back to color fill")
            return self._fallback_color_fill(img_gpu, mask_gpu)

    def _fallback_color_fill(self, img_gpu: torch.Tensor, mask_gpu: torch.Tensor) -> torch.Tensor:
        """Fallback: заливка цветом окружения если LaMa недоступен."""
        h, w = img_gpu.shape[:2]
        processed = img_gpu.clone().float()

        # Находим связные компоненты в маске
        mask_np = (mask_gpu > 0.5).cpu().numpy().astype(np.uint8)

        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(mask_np)

            for i in range(1, num_features + 1):
                coords = np.where(labeled == i)
                if len(coords[0]) == 0:
                    continue

                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()

                color = _get_surrounding_color_gpu(img_gpu.float(), x1, y1, x2, y2, self.SURROUNDING_MARGIN)
                processed[y1:y2+1, x1:x2+1, :3] = color

        except ImportError:
            # Если scipy недоступен, просто заливаем всю маску средним цветом
            mask_3d = mask_gpu.unsqueeze(-1).expand(-1, -1, 3)
            mean_color = img_gpu[mask_gpu < 0.5].reshape(-1, 3).float().mean(dim=0)
            processed = torch.where(mask_3d > 0.5, mean_color, processed)

        return (processed / 255.0).clamp(0, 1)


class ImageSwitch:
    """
    Переключатель изображений: если первое изображение полностью белое/чёрное,
    на выход идёт второе изображение. Иначе — первое.

    Полезно для условной логики в workflow:
    - Если обработка вернула пустой результат → использовать оригинал
    - Если маска пустая → пропустить шаг
    """

    # Пороги для определения "пустого" изображения
    BLACK_THRESHOLD = 5           # Среднее значение < 5 = чёрное
    WHITE_THRESHOLD = 250         # Среднее значение > 250 = белое
    UNIFORMITY_THRESHOLD = 3.0    # STD < 3 = однотонное

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_primary": ("IMAGE",),
                "image_fallback": ("IMAGE",),
            },
            "optional": {
                "check_uniform": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "used_fallback")
    FUNCTION = "switch"
    CATEGORY = "image/preprocessing"

    def switch(self, image_primary: torch.Tensor, image_fallback: torch.Tensor, check_uniform: bool = True):
        """
        Проверяет первое изображение и возвращает:
        - image_primary если оно НЕ пустое (не чисто белое/чёрное)
        - image_fallback если image_primary пустое

        Args:
            image_primary: Основное изображение [B, H, W, C]
            image_fallback: Запасное изображение [B, H, W, C]
            check_uniform: Проверять ли однотонность (не только чёрное/белое)

        Returns:
            (image, used_fallback): Выбранное изображение и флаг использования fallback
        """
        batch_size = image_primary.shape[0]
        results = []
        used_fallback_flags = []

        for i in range(batch_size):
            img = image_primary[i]
            is_empty = self._is_empty_image(img, check_uniform)

            if is_empty:
                # Используем fallback
                if i < image_fallback.shape[0]:
                    results.append(image_fallback[i])
                else:
                    results.append(image_fallback[0])  # Если батч не совпадает
                used_fallback_flags.append(True)
                logger.info(f"ImageSwitch: image {i} is empty, using fallback")
            else:
                results.append(img)
                used_fallback_flags.append(False)

        out_image = torch.stack(results)
        # Возвращаем True если хотя бы один fallback использован
        used_fallback = any(used_fallback_flags)

        return (out_image, used_fallback)

    def _is_empty_image(self, img: torch.Tensor, check_uniform: bool) -> bool:
        """Проверяет, является ли изображение пустым (белым/чёрным/однотонным)."""
        # Конвертируем в 0-255 для проверки
        img_255 = (img.clamp(0, 1) * 255).float()

        # Среднее значение
        mean_val = img_255.mean().item()

        # Проверка на чёрное
        if mean_val < self.BLACK_THRESHOLD:
            logger.debug(f"Image is black: mean={mean_val:.1f}")
            return True

        # Проверка на белое
        if mean_val > self.WHITE_THRESHOLD:
            logger.debug(f"Image is white: mean={mean_val:.1f}")
            return True

        # Проверка на однотонность (опционально)
        if check_uniform:
            std_val = img_255.std().item()
            if std_val < self.UNIFORMITY_THRESHOLD:
                logger.debug(f"Image is uniform: std={std_val:.1f}")
                return True

        return False



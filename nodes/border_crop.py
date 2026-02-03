import torch
import numpy as np
import os
import logging
import cv2

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


class AutoBorderCrop:
    """Обрезает однотонные рамки с краёв изображения."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sensitivity": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "min_border_size": ("INT", {"default": 5, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_borders"
    CATEGORY = "image/preprocessing"

    def crop_borders(self, image: torch.Tensor, sensitivity: float = 15.0, min_border_size: int = 5):
        # Берём только первый кадр
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img.shape[:2]

        crop = {}
        for side in ['top', 'bottom', 'left', 'right']:
            crop[side] = self._detect_border(img, side, sensitivity, min_border_size)

        if (h - crop['top'] - crop['bottom']) < h * 0.5 or (w - crop['left'] - crop['right']) < w * 0.5:
            return (image[0:1],)

        y1, y2 = crop['top'], h - crop['bottom'] if crop['bottom'] > 0 else h
        x1, x2 = crop['left'], w - crop['right'] if crop['right'] > 0 else w

        return (image[0:1, y1:y2, x1:x2, :],)

    def _is_grayscale_color(self, rgb: np.ndarray, saturation_threshold: float = 30.0) -> bool:
        """Проверяет, является ли цвет оттенком серого (чёрный/белый/серый)."""
        r, g, b = rgb[0], rgb[1], rgb[2]
        max_val = max(r, g, b)
        min_val = min(r, g, b)

        # Насыщенность (разница между макс и мин каналами)
        saturation = max_val - min_val
        return saturation < saturation_threshold

    def _detect_border(self, img, side, sensitivity, min_size):
        h, w = img.shape[:2]

        if side == 'top':
            get_line = lambda i: img[i, :, :3]
            max_scan = h // 2
        elif side == 'bottom':
            get_line = lambda i: img[h-1-i, :, :3]
            max_scan = h // 2
        elif side == 'left':
            get_line = lambda i: img[:, i, :3]
            max_scan = w // 2
        else:
            get_line = lambda i: img[:, w-1-i, :3]
            max_scan = w // 2

        # Берём угловой сэмпл для определения цвета рамки
        corner_size = min(50, h // 4, w // 4)
        if side in ['top', 'bottom']:
            corner = img[0:corner_size, 0:corner_size, :3] if side == 'top' else img[h-corner_size:h, 0:corner_size, :3]
        else:
            corner = img[0:corner_size, 0:corner_size, :3] if side == 'left' else img[0:corner_size, w-corner_size:w, :3]

        ref_color = np.median(corner.reshape(-1, 3).astype(np.float32), axis=0)

        # Проверяем что это оттенок серого
        if not self._is_grayscale_color(ref_color):
            return 0

        border = 0
        for i in range(max_scan):
            line = get_line(i).astype(np.float32)
            line_median = np.median(line, axis=0)

            # Проверяем что линия похожа на рамку (серая и близка к ref)
            if self._is_grayscale_color(line_median):
                if np.abs(line_median - ref_color).mean() < sensitivity:
                    border = i + 1
                elif np.std(line, axis=0).mean() > sensitivity * 3:
                    # Линия неоднородная — конец рамки
                    break
                else:
                    break
            else:
                break

        return border if border >= min_size else 0


class SmartScreenshotCleaner:
    """
    Находит UI элементы и заливает их цветом окружения.
    Если ничего не найдено — пропускает без обработки.
    """

    yolo_model = None  # Singleton

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.9, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "ui_mask")
    FUNCTION = "process"
    CATEGORY = "image/preprocessing"

    def __init__(self):
        self.device = get_device()

    def _get_models_dir(self):
        try:
            import folder_paths
            d = os.path.join(folder_paths.models_dir, "screenshot_cleaner")
        except ImportError:
            d = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(d, exist_ok=True)
        return d

    def _load_yolo(self):
        if SmartScreenshotCleaner.yolo_model is not None:
            return SmartScreenshotCleaner.yolo_model
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect/model.pt", cache_dir=self._get_models_dir())
            SmartScreenshotCleaner.yolo_model = YOLO(path)
            if self.device.type == "cuda":
                SmartScreenshotCleaner.yolo_model.to(self.device)
            logger.info("YOLO loaded")
            return SmartScreenshotCleaner.yolo_model
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            return None

    def _get_surrounding_color(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin: int = 5) -> np.ndarray:
        """Получает средний цвет пикселей вокруг области."""
        h, w = img.shape[:2]

        # Собираем пиксели вокруг бокса
        pixels = []

        # Сверху
        if y1 > 0:
            top = max(0, y1 - margin)
            pixels.append(img[top:y1, x1:x2].reshape(-1, 3))

        # Снизу
        if y2 < h:
            bottom = min(h, y2 + margin)
            pixels.append(img[y2:bottom, x1:x2].reshape(-1, 3))

        # Слева
        if x1 > 0:
            left = max(0, x1 - margin)
            pixels.append(img[y1:y2, left:x1].reshape(-1, 3))

        # Справа
        if x2 < w:
            right = min(w, x2 + margin)
            pixels.append(img[y1:y2, x2:right].reshape(-1, 3))

        if pixels:
            all_pixels = np.concatenate(pixels, axis=0)
            return np.median(all_pixels, axis=0).astype(np.uint8)

        return np.array([128, 128, 128], dtype=np.uint8)

    def _is_uniform_background(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, threshold: float = 25.0) -> bool:
        """Проверяет однотонность фона вокруг области."""
        h, w = img.shape[:2]
        margin = 10

        pixels = []
        if y1 > margin:
            pixels.append(img[y1-margin:y1, x1:x2].reshape(-1, 3))
        if y2 < h - margin:
            pixels.append(img[y2:y2+margin, x1:x2].reshape(-1, 3))
        if x1 > margin:
            pixels.append(img[y1:y2, x1-margin:x1].reshape(-1, 3))
        if x2 < w - margin:
            pixels.append(img[y1:y2, x2:x2+margin].reshape(-1, 3))

        if not pixels:
            return False

        all_pixels = np.concatenate(pixels, axis=0).astype(np.float32)
        std = np.std(all_pixels, axis=0).mean()
        return std < threshold

    @torch.no_grad()
    def process(self, image: torch.Tensor, confidence: float = 0.2):
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]

        model = self._load_yolo()

        # Если модель не загружена — возвращаем как есть
        if model is None:
            return (image, torch.zeros(batch_size, h, w))

        # Только первый кадр обрабатываем
        img = image[0].cpu().numpy()
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Детекция с уменьшенным размером для скорости
        results = model.predict(img_uint8, conf=confidence, verbose=False, device=self.device, imgsz=640)

        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

        # Если ничего не найдено — возвращаем как есть (быстро)
        if not boxes:
            return (image[0:1], torch.zeros(1, h, w))

        logger.info(f"UI elements: {len(boxes)}")

        # Заливка
        ui_mask = np.zeros((h, w), dtype=np.float32)
        processed = img_uint8.copy()

        for x1, y1, x2, y2 in boxes:
            ui_mask[y1:y2, x1:x2] = 1.0
            if self._is_uniform_background(img_uint8, x1, y1, x2, y2):
                color = self._get_surrounding_color(img_uint8, x1, y1, x2, y2)
                processed[y1:y2, x1:x2] = color

        out_img = torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0)
        out_mask = torch.from_numpy(ui_mask).unsqueeze(0)

        return (out_img, out_mask)

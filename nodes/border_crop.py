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

        ref_color = np.median(get_line(0).astype(np.float32), axis=0)
        if np.std(get_line(0), axis=0).mean() > sensitivity * 2:
            return 0

        border = 0
        for i in range(max_scan):
            line = get_line(i).astype(np.float32)
            if np.abs(np.mean(line, axis=0) - ref_color).mean() < sensitivity:
                border = i + 1
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
        result_images = []
        result_masks = []

        model = self._load_yolo()

        for b in range(batch_size):
            img = image[b].cpu().numpy()
            h, w = img.shape[:2]
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            # Пустая маска
            ui_mask = np.zeros((h, w), dtype=np.float32)

            # Если модель не загружена — пропускаем
            if model is None:
                result_images.append(image[b])
                result_masks.append(torch.zeros(h, w))
                continue

            # Детекция
            results = model.predict(img_uint8, conf=confidence, verbose=False, device=self.device)

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

            # Если ничего не найдено — возвращаем оригинал
            if not boxes:
                result_images.append(image[b])
                result_masks.append(torch.zeros(h, w))
                continue

            logger.info(f"UI elements: {len(boxes)}")

            # Заливка каждого бокса
            processed = img_uint8.copy()
            for x1, y1, x2, y2 in boxes:
                ui_mask[y1:y2, x1:x2] = 1.0

                # Если фон однотонный — заливаем цветом (быстро)
                if self._is_uniform_background(img_uint8, x1, y1, x2, y2):
                    color = self._get_surrounding_color(img_uint8, x1, y1, x2, y2)
                    processed[y1:y2, x1:x2] = color

            result_images.append(torch.from_numpy(processed.astype(np.float32) / 255.0))
            result_masks.append(torch.from_numpy(ui_mask))

        out_img = torch.stack(result_images, dim=0)
        out_mask = torch.stack(result_masks, dim=0)

        return (out_img, out_mask)

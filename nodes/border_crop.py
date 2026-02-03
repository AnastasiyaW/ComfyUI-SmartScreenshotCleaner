import torch
import numpy as np
from typing import Tuple, Optional
import os
import logging
import cv2

logger = logging.getLogger("ScreenshotCleaner")
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
    """Обрезает однотонные рамки (тёмные/светлые) с краёв изображения."""

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
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img = (image[b].cpu().numpy() * 255).astype(np.uint8)
            h, w = img.shape[:2]

            crop = {}
            for side in ['top', 'bottom', 'left', 'right']:
                crop[side] = self._detect_border(img, side, sensitivity, min_border_size)

            # Проверка что осталось >50% контента
            if (h - crop['top'] - crop['bottom']) < h * 0.5 or (w - crop['left'] - crop['right']) < w * 0.5:
                crop = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

            y1, y2 = crop['top'], h - crop['bottom'] if crop['bottom'] > 0 else h
            x1, x2 = crop['left'], w - crop['right'] if crop['right'] > 0 else w
            results.append(image[b:b+1, y1:y2, x1:x2, :])

        return (torch.cat(results, dim=0),)

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
    Находит UI элементы на скриншоте и замазывает их в цвет окружающих пикселей.

    Использует YOLO (OmniParser) для детекции UI, затем LaMa для инпеинтинга.
    НЕ обрезает изображение — только замазывает UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand_mask": ("INT", {"default": 5, "min": 0, "max": 30, "tooltip": "Расширить маску UI для лучшей заливки"}),
                "confidence": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.9, "step": 0.05, "tooltip": "Порог уверенности YOLO"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "ui_mask")
    FUNCTION = "process"
    CATEGORY = "image/preprocessing"

    def __init__(self):
        self.device = get_device()
        self.lama_model = None
        self.yolo_model = None
        self._models_dir = self._get_models_dir()

    def _get_models_dir(self):
        try:
            import folder_paths
            d = os.path.join(folder_paths.models_dir, "screenshot_cleaner")
        except ImportError:
            d = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(d, exist_ok=True)
        return d

    def _load_yolo(self):
        if self.yolo_model:
            return self.yolo_model
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect/model.pt", cache_dir=self._models_dir)
            self.yolo_model = YOLO(path)
            if self.device.type == "cuda":
                self.yolo_model.to(self.device)
            logger.info("YOLO (OmniParser) loaded")
            return self.yolo_model
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            return None

    def _load_lama(self):
        if self.lama_model:
            return self.lama_model
        try:
            from simple_lama_inpainting import SimpleLama
            self.lama_model = SimpleLama()
            logger.info("LaMa loaded")
            return self.lama_model
        except Exception as e:
            logger.warning(f"LaMa not available: {e}")
            return None

    @torch.no_grad()
    def _get_ui_mask(self, img_np: np.ndarray, confidence: float) -> np.ndarray:
        """Детектирует UI элементы через YOLO."""
        model = self._load_yolo()
        if not model:
            return np.zeros(img_np.shape[:2], dtype=np.float32)

        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        results = model.predict(img_np, conf=confidence, verbose=False, device=self.device)

        count = 0
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box

                # Проверка валидности координат
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 1.0
                    count += 1

        logger.info(f"UI elements detected: {count}")
        return mask

    @torch.no_grad()
    def _inpaint(self, img_np: np.ndarray, mask: np.ndarray, expand_px: int) -> np.ndarray:
        """Закрашивает UI через LaMa или OpenCV fallback."""
        from PIL import Image

        if mask.max() == 0:
            return img_np

        # Расширяем маску
        if expand_px > 0:
            kernel = np.ones((expand_px*2+1, expand_px*2+1), np.uint8)
            mask_uint8 = cv2.dilate((mask * 255).astype(np.uint8), kernel)
        else:
            mask_uint8 = (mask * 255).astype(np.uint8)

        # Пробуем LaMa
        lama = self._load_lama()
        if lama:
            try:
                result = lama(Image.fromarray(img_np), Image.fromarray(mask_uint8))
                return np.array(result)
            except Exception as e:
                logger.warning(f"LaMa error: {e}")

        # Fallback на OpenCV inpaint
        logger.info("Using OpenCV inpaint fallback")
        return cv2.inpaint(img_np, mask_uint8, 3, cv2.INPAINT_TELEA)

    @torch.no_grad()
    def process(self, image: torch.Tensor, expand_mask: int = 5, confidence: float = 0.2):
        batch_size = image.shape[0]
        result_images = []
        result_masks = []

        for b in range(batch_size):
            img = image[b].cpu().numpy()
            h, w = img.shape[:2]
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            # 1. Детектируем UI элементы
            ui_mask = self._get_ui_mask(img_uint8, confidence)

            # 2. Инпеинтим UI
            if ui_mask.max() > 0:
                processed = self._inpaint(img_uint8, ui_mask, expand_mask)
            else:
                processed = img_uint8

            result_images.append(torch.from_numpy(processed.astype(np.float32) / 255.0))
            result_masks.append(torch.from_numpy(ui_mask))

        out_img = torch.stack(result_images, dim=0)
        out_mask = torch.stack(result_masks, dim=0)

        logger.info(f"Output shape: {out_img.shape}")
        return (out_img, out_mask)

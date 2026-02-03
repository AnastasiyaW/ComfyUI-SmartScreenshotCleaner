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
    """Обрезает однотонные чёрные/белые/серые рамки с краёв изображения."""

    SAFE_MARGIN = 50  # Минимум пикселей от главного объекта

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
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img.shape[:2]
        need_lama = False

        # Находим bbox главного объекта
        content_bbox = self._find_content_bbox(img)
        logger.info(f"Content bbox: {content_bbox}")

        # Детектируем рамки слева и справа
        crop = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        crop['left'] = self._detect_border(img, 'left', sensitivity, min_border_size)
        crop['right'] = self._detect_border(img, 'right', sensitivity, min_border_size)

        # Сверху и снизу — определяем тип краёв
        top_type = self._analyze_edge_strip(img, 'top')
        bottom_type = self._analyze_edge_strip(img, 'bottom')

        logger.info(f"Edge types: top={top_type}, bottom={bottom_type}")
        logger.info(f"Detected borders: left={crop['left']}, right={crop['right']}")

        top_border = self._detect_border(img, 'top', sensitivity, min_border_size)
        bottom_border = self._detect_border(img, 'bottom', sensitivity, min_border_size)

        if content_bbox:
            cy1, cy2, cx1, cx2 = content_bbox
            safe_top = max(0, cy1 - self.SAFE_MARGIN)
            safe_bottom = max(0, h - cy2 - self.SAFE_MARGIN)

            # Сверху
            if top_border > safe_top:
                if top_type == 'uniform':
                    # Заливаем safe zone цветом фона
                    fill_color = self._get_border_color(img, 'top')
                    img[safe_top:top_border, :] = fill_color
                    crop['top'] = top_border
                    logger.info(f"Top: uniform, filled {safe_top}-{top_border} with color, crop={top_border}")
                elif top_type == 'dynamic':
                    crop['top'] = top_border
                    logger.info(f"Top: dynamic edge, full crop={top_border}")
                else:
                    # Mixed — нужен LaMa
                    need_lama = True
                    crop['top'] = safe_top
                    logger.info(f"Top: mixed, crop to safe={safe_top}, LaMa needed")
            else:
                crop['top'] = top_border

            # Снизу
            if bottom_border > safe_bottom:
                if bottom_type == 'uniform':
                    fill_color = self._get_border_color(img, 'bottom')
                    img[h-bottom_border:h-safe_bottom, :] = fill_color
                    crop['bottom'] = bottom_border
                    logger.info(f"Bottom: uniform, filled, crop={bottom_border}")
                elif bottom_type == 'dynamic':
                    crop['bottom'] = bottom_border
                    logger.info(f"Bottom: dynamic edge, full crop={bottom_border}")
                else:
                    need_lama = True
                    crop['bottom'] = safe_bottom
                    logger.info(f"Bottom: mixed, crop to safe={safe_bottom}, LaMa needed")
            else:
                crop['bottom'] = bottom_border
        else:
            crop['top'] = top_border
            crop['bottom'] = bottom_border

        # Применяем LaMa если нужно
        if need_lama and content_bbox:
            img = self._apply_lama_to_borders(img, crop, content_bbox, top_type, bottom_type)

        logger.info(f"Final crop: top={crop['top']}, bottom={crop['bottom']}, left={crop['left']}, right={crop['right']}")

        # Проверка что осталось достаточно контента
        if (h - crop['top'] - crop['bottom']) < h * 0.3 or (w - crop['left'] - crop['right']) < w * 0.3:
            return (image[0:1],)

        y1, y2 = crop['top'], h - crop['bottom'] if crop['bottom'] > 0 else h
        x1, x2 = crop['left'], w - crop['right'] if crop['right'] > 0 else w

        if y1 >= y2 or x1 >= x2:
            return (image[0:1],)

        # Возвращаем обработанное изображение
        result = torch.from_numpy(img[y1:y2, x1:x2, :].astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)

    def _get_border_color(self, img: np.ndarray, side: str) -> np.ndarray:
        """Получает цвет рамки для заливки."""
        h, w = img.shape[:2]
        if side == 'top':
            sample = img[0:10, :, :3]
        else:
            sample = img[h-10:h, :, :3]

        median = np.median(sample.reshape(-1, 3), axis=0)

        if median.mean() < 40:
            return np.array([0, 0, 0], dtype=np.uint8)
        elif median.mean() > 215:
            return np.array([255, 255, 255], dtype=np.uint8)
        return median.astype(np.uint8)

    def _apply_lama_to_borders(self, img: np.ndarray, crop: dict, content_bbox: tuple, top_type: str, bottom_type: str) -> np.ndarray:
        """Применяет LaMa для mixed областей."""
        try:
            from simple_lama_inpainting import SimpleLama
            from PIL import Image

            h, w = img.shape[:2]
            cy1, cy2, cx1, cx2 = content_bbox
            mask = np.zeros((h, w), dtype=np.uint8)

            # Маска для mixed областей
            if top_type == 'mixed':
                top_border = self._detect_border(img, 'top', 15, 5)
                safe_top = max(0, cy1 - self.SAFE_MARGIN)
                if top_border > safe_top:
                    mask[safe_top:top_border, :] = 255

            if bottom_type == 'mixed':
                bottom_border = self._detect_border(img, 'bottom', 15, 5)
                safe_bottom = max(0, h - cy2 - self.SAFE_MARGIN)
                if bottom_border > safe_bottom:
                    mask[h-bottom_border:h-safe_bottom, :] = 255

            if mask.max() == 0:
                return img

            logger.info("Applying LaMa for mixed border areas")
            lama = SimpleLama()
            result = lama(Image.fromarray(img), Image.fromarray(mask))
            return np.array(result)

        except Exception as e:
            logger.warning(f"LaMa failed: {e}")
            return img

    def _analyze_edge_strip(self, img: np.ndarray, side: str) -> str:
        """Анализирует полосу с края: uniform, dynamic, mixed."""
        h, w = img.shape[:2]
        edge_size = 30

        if side == 'top':
            strip = img[0:edge_size, :, :3]
        elif side == 'bottom':
            strip = img[h-edge_size:h, :, :3]
        elif side == 'left':
            strip = img[:, 0:edge_size, :3]
        else:
            strip = img[:, w-edge_size:w, :3]

        pixels = strip.reshape(-1, 3).astype(np.float32)
        std = np.std(pixels, axis=0).mean()
        median = np.median(pixels, axis=0)
        saturation = max(median) - min(median)

        if std < 15 and saturation < 30:
            return 'uniform'
        elif std > 40:
            return 'dynamic'
        else:
            return 'mixed'

    def _find_content_bbox(self, img: np.ndarray) -> tuple:
        """Находит bbox контента (не чёрное/белое/серое)."""
        h, w = img.shape[:2]

        # Конвертируем в grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Маска: не слишком тёмное и не слишком светлое
        # Чёрное < 30, белое > 225
        content_mask = ((gray > 30) & (gray < 225)).astype(np.uint8)

        # Также проверяем насыщенность (цветное = контент)
        if len(img.shape) == 3:
            max_ch = np.max(img, axis=2)
            min_ch = np.min(img, axis=2)
            saturation = max_ch.astype(np.int16) - min_ch.astype(np.int16)
            color_mask = (saturation > 20).astype(np.uint8)
            content_mask = np.maximum(content_mask, color_mask)

        # Морфология для удаления шума
        kernel = np.ones((5, 5), np.uint8)
        content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
        content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)

        # Находим bbox контента
        coords = np.where(content_mask > 0)
        if len(coords[0]) == 0:
            return None

        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()

        return (y1, y2, x1, x2)

    def _is_grayscale_color(self, rgb: np.ndarray, saturation_threshold: float = 30.0) -> bool:
        """Проверяет, является ли цвет оттенком серого."""
        r, g, b = rgb[0], rgb[1], rgb[2]
        saturation = max(r, g, b) - min(r, g, b)
        return saturation < saturation_threshold

    def _detect_border(self, img, side, sensitivity, min_size):
        """Двойной проход: сначала полосы, потом углы."""
        h, w = img.shape[:2]

        # Проход 1: полосы по всей стороне
        border_pass1 = self._detect_border_by_lines(img, side, sensitivity, min_size)

        # Проход 2: проверка угла после первого прохода
        border_pass2 = self._detect_border_by_corner(img, side, sensitivity, min_size, border_pass1)

        return max(border_pass1, border_pass2)

    def _detect_border_by_lines(self, img, side, sensitivity, min_size):
        """Проход 1: детекция по полным линиям."""
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

        # Референсный цвет из первых линий
        sample_lines = min(5, max_scan)
        sample_pixels = []
        for i in range(sample_lines):
            sample_pixels.append(get_line(i))
        sample = np.vstack(sample_pixels).astype(np.float32)
        ref_color = np.median(sample, axis=0)

        if not self._is_grayscale_color(ref_color):
            return 0

        border = 0
        for i in range(max_scan):
            line = get_line(i).astype(np.float32)
            line_median = np.median(line, axis=0)

            if self._is_grayscale_color(line_median):
                diff = np.abs(line_median - ref_color).mean()
                if diff < sensitivity:
                    border = i + 1
                else:
                    break
            else:
                break

        return border if border >= min_size else 0

    def _detect_border_by_corner(self, img, side, sensitivity, min_size, start_offset):
        """Проход 2: проверка угла — если там ещё серое, дорезаем."""
        h, w = img.shape[:2]
        corner_size = min(100, h // 4, w // 4)

        # Берём угол с учётом уже срезанного
        if side == 'top':
            y_start = start_offset
            corner = img[y_start:y_start+corner_size, 0:corner_size, :3]
            get_line = lambda i: img[start_offset + i, :, :3]
            max_scan = h // 2 - start_offset
        elif side == 'bottom':
            y_start = h - start_offset - corner_size
            corner = img[max(0, y_start):h-start_offset, 0:corner_size, :3]
            get_line = lambda i: img[h-1-start_offset-i, :, :3]
            max_scan = h // 2 - start_offset
        elif side == 'left':
            x_start = start_offset
            corner = img[0:corner_size, x_start:x_start+corner_size, :3]
            get_line = lambda i: img[:, start_offset + i, :3]
            max_scan = w // 2 - start_offset
        else:
            x_start = w - start_offset - corner_size
            corner = img[0:corner_size, max(0, x_start):w-start_offset, :3]
            get_line = lambda i: img[:, w-1-start_offset-i, :3]
            max_scan = w // 2 - start_offset

        if corner.size == 0 or max_scan <= 0:
            return 0

        ref_color = np.median(corner.reshape(-1, 3).astype(np.float32), axis=0)

        if not self._is_grayscale_color(ref_color):
            return 0

        extra_border = 0
        for i in range(max_scan):
            line = get_line(i).astype(np.float32)
            line_median = np.median(line, axis=0)

            if self._is_grayscale_color(line_median):
                diff = np.abs(line_median - ref_color).mean()
                if diff < sensitivity * 1.5:  # Чуть мягче для второго прохода
                    extra_border = i + 1
                else:
                    break
            else:
                break

        total = start_offset + extra_border
        return total if extra_border >= min_size else start_offset


class SmartScreenshotCleaner:
    """
    Находит UI элементы и заливает их цветом окружения.
    Если ничего не найдено — пропускает без обработки.
    """

    yolo_model = None

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

    def _get_surrounding_color(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin: int = 10) -> np.ndarray:
        """Получает цвет пикселей вокруг области. Если чёрное — чёрный, белое — белый."""
        h, w = img.shape[:2]
        pixels = []

        if y1 > 0:
            pixels.append(img[max(0, y1-margin):y1, x1:x2].reshape(-1, 3))
        if y2 < h:
            pixels.append(img[y2:min(h, y2+margin), x1:x2].reshape(-1, 3))
        if x1 > 0:
            pixels.append(img[y1:y2, max(0, x1-margin):x1].reshape(-1, 3))
        if x2 < w:
            pixels.append(img[y1:y2, x2:min(w, x2+margin)].reshape(-1, 3))

        if not pixels:
            return np.array([128, 128, 128], dtype=np.uint8)

        all_pixels = np.concatenate(pixels, axis=0)
        median_color = np.median(all_pixels, axis=0)

        # Если почти чёрное (< 40) — делаем чисто чёрным
        if median_color.mean() < 40:
            return np.array([0, 0, 0], dtype=np.uint8)

        # Если почти белое (> 215) — делаем чисто белым
        if median_color.mean() > 215:
            return np.array([255, 255, 255], dtype=np.uint8)

        return median_color.astype(np.uint8)

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
        h, w = image.shape[1], image.shape[2]

        model = self._load_yolo()
        if model is None:
            return (image[0:1], torch.zeros(1, h, w))

        img = image[0].cpu().numpy()
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

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

        if not boxes:
            return (image[0:1], torch.zeros(1, h, w))

        logger.info(f"UI elements: {len(boxes)}")

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

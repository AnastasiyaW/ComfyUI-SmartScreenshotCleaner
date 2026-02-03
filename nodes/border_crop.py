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
    Использует YOLO для детекции людей и построения safe bbox."""

    # Константы класса — Safe margins
    SAFE_MARGIN_VERTICAL = 50      # Отступ сверху/снизу от контента
    SAFE_MARGIN_HORIZONTAL = 200   # Большой отступ слева/справа (защита рук)

    # Константы — Edge detection
    EDGE_SIZE = 30
    CORNER_SIZE_RATIO = 4
    MAX_CORNER_SIZE = 100
    CHECK_DEPTH = 20               # Глубина проверки границы (пикселей)
    LOOKAHEAD_PIXELS = 30          # Смотрим вперёд для проверки контента

    # Константы — Thresholds
    UNIFORM_STD_THRESHOLD = 15.0   # Ниже = чистый однотонный фон
    AMBIGUOUS_STD_LOW = 15.0       # Нижний порог неоднозначной зоны
    AMBIGUOUS_STD_HIGH = 40.0      # Верхний порог неоднозначной зоны
    DYNAMIC_STD_THRESHOLD = 50.0   # Выше = высокая динамика (контент)
    CONTENT_STD_THRESHOLD = 40.0   # Порог для проверки контента
    SEGMENT_STD_THRESHOLD = 20.0   # Порог динамики сегмента линии

    # Константы — Color detection
    SATURATION_THRESHOLD = 30.0    # Порог насыщенности для цветного контента
    BLACK_THRESHOLD = 30           # Ниже = чёрный фон
    WHITE_THRESHOLD = 225          # Выше = белый фон
    COLOR_SATURATION_THRESHOLD = 20
    COLORFUL_RATIO_THRESHOLD = 0.2 # 20% цветных пикселей = цветной контент
    HIGH_DYNAMICS_STD = 50.0       # STD для игнорирования safe space

    # Константы — Limits
    MIN_CONTENT_RATIO = 0.3        # Минимум 30% контента после обрезки
    MIN_COVERAGE = 0.7             # Минимум 70% покрытия для контента
    CONTENT_CONTINUE_RATIO = 0.6   # 60% линий для подтверждения контента
    SECOND_PASS_SENSITIVITY_MULTIPLIER = 1.5

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

    @torch.no_grad()
    def crop_borders(self, image: torch.Tensor, sensitivity: float = 15.0, min_border_size: int = 5):
        """
        Обрезает однотонные рамки с краёв изображения.

        Args:
            image: Тензор изображения формата [B, H, W, C] со значениями в [0, 1]
            sensitivity: Чувствительность детекции рамки (1.0-50.0)
            min_border_size: Минимальный размер рамки для обрезки в пикселях

        Returns:
            Tuple[torch.Tensor]: Обрезанное изображение формата [B, H', W', C]
        """
        # Валидация входных данных
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image)}")
        if len(image.shape) != 4:
            raise ValueError(f"Expected image shape [B, H, W, C], got {image.shape}")
        if image.shape[1] < 10 or image.shape[2] < 10:
            logger.warning(f"Image too small: {image.shape}, returning original")
            return (image,)

        device = get_device()
        batch_size = image.shape[0]
        results = []

        for batch_idx in range(batch_size):
            single_image = image[batch_idx]
            result = self._process_single_image_gpu(single_image, sensitivity, min_border_size, device)
            results.append(result)

        if len(results) == 1:
            return (results[0].unsqueeze(0),)

        max_h = max(r.shape[0] for r in results)
        max_w = max(r.shape[1] for r in results)

        padded_results = []
        for r in results:
            h, w = r.shape[:2]
            if h < max_h or w < max_w:
                pad_h = max_h - h
                pad_w = max_w - w
                r = F.pad(r.permute(2, 0, 1), (0, pad_w, 0, pad_h), mode='constant', value=0).permute(1, 2, 0)
            padded_results.append(r)

        return (torch.stack(padded_results),)

    def _detect_person_bbox_yolo(self, img_gpu: torch.Tensor, device: torch.device) -> Optional[Tuple[int, int, int, int]]:
        """Детектирует человека через YOLO v11 и возвращает bbox."""
        try:
            model = _load_yolo_model("yolo_person", device)
            if model is None:
                return None

            # Конвертируем для YOLO
            img_np = img_gpu.cpu().numpy().astype(np.uint8)

            # Инференс
            results = model.predict(img_np, conf=0.3, verbose=False, device=device, imgsz=640, classes=[0])  # class 0 = person

            if not results or len(results) == 0:
                return None

            # Находим самый большой bbox человека
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

    def _find_sharp_content_edge(self, img: torch.Tensor, side: str, search_start: int, search_end: int) -> Tuple[Optional[int], Optional[int]]:
        """Ищет чёткую границу контента (резкий переход от фона к контенту по всей длине линии).

        Возвращает (sharp_edge, ambiguous_start):
        - sharp_edge: позиция чёткой границы или None
        - ambiguous_start: начало неоднозначной зоны (для LaMa) или None

        Логика:
        1. Чёткая граница = резкий скачок динамики пикселей по всей длине линии
        2. Неоднозначная зона = переход (тень, градиент) между фоном и контентом
        3. Если за неоднозначной зоной есть реальный объект — не режем, а LaMa заливает тень
        """
        h, w = img.shape[:2]

        if search_start >= search_end:
            return None, None

        # Определяем направление сканирования
        if side == 'top':
            get_line = lambda i: img[i, :, :3]
            max_pos = h
        elif side == 'bottom':
            get_line = lambda i: img[h - 1 - i, :, :3]
            max_pos = h
        elif side == 'left':
            get_line = lambda i: img[:, i, :3]
            max_pos = w
        else:  # right
            get_line = lambda i: img[:, w - 1 - i, :3]
            max_pos = w

        prev_is_uniform = True
        ambiguous_start = None
        in_ambiguous_zone = False

        for pos in range(search_start, min(search_end, max_pos)):
            line = get_line(pos).float()
            line_std = line.std(dim=0).mean().item()

            # Проверяем яркость — чёрный/белый фон?
            line_mean = line.mean().item()
            is_bw_background = line_mean < self.BLACK_THRESHOLD or line_mean > self.WHITE_THRESHOLD

            # Проверяем покрытие динамикой
            num_segments = 10
            if side in ('top', 'bottom'):
                segment_size = max(1, w // num_segments)
                segments_dynamic = 0
                for seg in range(num_segments):
                    seg_start = seg * segment_size
                    seg_end = min((seg + 1) * segment_size, w)
                    seg_pixels = line[seg_start:seg_end]
                    if seg_pixels.numel() > 0:
                        seg_std = seg_pixels.std(dim=0).mean().item()
                        if seg_std > self.SEGMENT_STD_THRESHOLD:
                            segments_dynamic += 1
            else:
                segment_size = max(1, h // num_segments)
                segments_dynamic = 0
                for seg in range(num_segments):
                    seg_start = seg * segment_size
                    seg_end = min((seg + 1) * segment_size, h)
                    seg_pixels = line[seg_start:seg_end]
                    if seg_pixels.numel() > 0:
                        seg_std = seg_pixels.std(dim=0).mean().item()
                        if seg_std > self.SEGMENT_STD_THRESHOLD:
                            segments_dynamic += 1

            coverage = segments_dynamic / num_segments
            is_content_line = line_std > self.DYNAMIC_STD_THRESHOLD and coverage > self.MIN_COVERAGE

            # Неоднозначная зона: не чистый фон, но и не контент (тень, градиент)
            is_ambiguous = (self.AMBIGUOUS_STD_LOW < line_std < self.AMBIGUOUS_STD_HIGH and
                           not is_bw_background and coverage < self.MIN_COVERAGE)

            # Запоминаем начало неоднозначной зоны
            if is_ambiguous and ambiguous_start is None and prev_is_uniform:
                ambiguous_start = pos
                in_ambiguous_zone = True
                logger.info(f"{side}: ambiguous zone (shadow/gradient) starts at {pos}, std={line_std:.1f}")

            # Нашли резкий переход к контенту
            if is_content_line:
                if in_ambiguous_zone:
                    # За тенью/градиентом есть реальный объект — это часть фото!
                    # Проверяем что объект продолжается дальше (не случайный шум)
                    content_continues = self._check_content_continues(img, side, pos, self.LOOKAHEAD_PIXELS)
                    if content_continues:
                        # Реальный объект за тенью — не режем! LaMa заполнит тень
                        logger.info(f"{side}: content found after shadow at {pos}, keeping object, LaMa will fill shadow")
                        return None, ambiguous_start
                    else:
                        # Случайный шум — режем до начала неоднозначной зоны
                        logger.info(f"{side}: found sharp edge at {pos} after ambiguous zone")
                        return pos, ambiguous_start
                elif prev_is_uniform:
                    # Чистый переход фон -> контент
                    logger.info(f"{side}: found sharp edge at {pos}, std={line_std:.1f}, coverage={coverage:.1%}")
                    return pos, None

            prev_is_uniform = line_std < self.AMBIGUOUS_STD_LOW
            if not is_ambiguous:
                in_ambiguous_zone = False

        # Не нашли чёткую границу, но может быть неоднозначная зона
        return None, ambiguous_start

    def _check_content_continues(self, img: torch.Tensor, side: str, start_pos: int, lookahead: int) -> bool:
        """Проверяет, продолжается ли контент дальше (не случайный шум).
        Если контент стабильно высокой динамики на протяжении lookahead пикселей — это реальный объект.
        """
        h, w = img.shape[:2]

        if side == 'top':
            get_line = lambda i: img[i, :, :3] if i < h else None
        elif side == 'bottom':
            get_line = lambda i: img[h - 1 - i, :, :3] if i < h else None
        elif side == 'left':
            get_line = lambda i: img[:, i, :3] if i < w else None
        else:
            get_line = lambda i: img[:, w - 1 - i, :3] if i < w else None

        content_lines = 0
        for offset in range(lookahead):
            line = get_line(start_pos + offset)
            if line is None:
                break
            line_std = line.float().std(dim=0).mean().item()
            if line_std > self.CONTENT_STD_THRESHOLD:
                content_lines += 1

        # Если больше CONTENT_CONTINUE_RATIO линий имеют высокую динамику — это реальный объект
        ratio = content_lines / max(1, lookahead)
        logger.info(f"{side}: content check at {start_pos}, {content_lines}/{lookahead} lines = {ratio:.1%}")
        return ratio > self.CONTENT_CONTINUE_RATIO

    def _is_pure_border_color(self, img: torch.Tensor, side: str, depth: int = 30) -> Tuple[bool, Optional[torch.Tensor]]:
        """Проверяет, является ли край чисто чёрным/белым/серым.
        Возвращает (is_pure, color) где color - медианный цвет края."""
        h, w = img.shape[:2]

        if side == 'left':
            strip = img[:, 0:depth, :3]
        elif side == 'right':
            strip = img[:, w-depth:w, :3]
        elif side == 'top':
            strip = img[0:depth, :, :3]
        else:  # bottom
            strip = img[h-depth:h, :, :3]

        pixels = strip.reshape(-1, 3).float()
        median = pixels.median(dim=0).values
        std = pixels.std(dim=0).mean().item()

        # Проверяем насыщенность (серый = низкая насыщенность)
        saturation = (median.max() - median.min()).item()
        mean_brightness = median.mean().item()

        # Чистый край: низкая std, низкая насыщенность, и либо тёмный либо светлый
        is_pure = (std < self.UNIFORM_STD_THRESHOLD and
                   saturation < self.SATURATION_THRESHOLD and
                   (mean_brightness < 50 or mean_brightness > 200))

        return is_pure, median

    def _check_content_at_edge(self, img: torch.Tensor, side: str, position: int) -> bool:
        """Проверяет, есть ли контент (не серый) на указанной позиции от края."""
        h, w = img.shape[:2]

        if side == 'left':
            strip = img[:, position:position+5, :3] if position < w else None
        elif side == 'right':
            pos = w - position - 5
            strip = img[:, max(0, pos):w-position, :3] if position < w else None
        elif side == 'top':
            strip = img[position:position+5, :, :3] if position < h else None
        else:  # bottom
            pos = h - position - 5
            strip = img[max(0, pos):h-position, :, :3] if position < h else None

        if strip is None or strip.numel() == 0:
            return False

        pixels = strip.reshape(-1, 3).float()

        # Проверяем насыщенность — если есть цветные пиксели, это контент
        max_ch = pixels.max(dim=1).values
        min_ch = pixels.min(dim=1).values
        saturation = max_ch - min_ch

        # Проверяем яркость — не чисто чёрный/белый
        brightness = pixels.mean(dim=1)

        # Контент = цветные пиксели ИЛИ средняя яркость (не чёрное/белое)
        color_content = (saturation > 30).float().mean().item()
        brightness_content = ((brightness > 40) & (brightness < 220)).float().mean().item()

        has_content = color_content > 0.1 or brightness_content > 0.3
        return has_content

    def _is_colorful_content_at_border(self, img: torch.Tensor, side: str, border_pos: int) -> bool:
        """Проверяет, есть ли ЦВЕТНОЙ контент на границе обрезки.

        Если цветной (не чёрно-белый/серый) — это точно не фон для обрезки,
        safe space не нужен, режем рамку полностью.

        Цветной = насыщенность > 30 (разница между max и min каналов RGB).
        Даже равномерное голубое небо — это цветной контент.
        """
        h, w = img.shape[:2]
        check_depth = self.CHECK_DEPTH

        if side == 'left':
            if border_pos >= w:
                return False
            strip = img[:, border_pos:min(border_pos + check_depth, w), :3]
        elif side == 'right':
            pos = w - border_pos
            if pos <= 0:
                return False
            strip = img[:, max(0, pos - check_depth):pos, :3]
        elif side == 'top':
            if border_pos >= h:
                return False
            strip = img[border_pos:min(border_pos + check_depth, h), :, :3]
        else:  # bottom
            pos = h - border_pos
            if pos <= 0:
                return False
            strip = img[max(0, pos - check_depth):pos, :, :3]

        if strip.numel() == 0:
            return False

        # Проверяем насыщенность (цветной контент)
        pixels = strip.reshape(-1, 3).float()
        saturation = (pixels.max(dim=1).values - pixels.min(dim=1).values)
        color_ratio = (saturation > self.SATURATION_THRESHOLD).float().mean().item()

        # Также проверяем высокую динамику — это тоже явный контент
        strip_std = strip.float().std(dim=(0, 1)).mean().item()

        # Цветной контент ИЛИ высокая динамика = не серый фон, safe space не нужен
        is_colorful = color_ratio > self.COLORFUL_RATIO_THRESHOLD or strip_std > self.HIGH_DYNAMICS_STD

        if is_colorful:
            logger.info(f"{side}: colorful/dynamic content at border {border_pos}, color={color_ratio:.1%}, std={strip_std:.1f} -> ignore safe space")

        return is_colorful

    def _process_single_image_gpu(self, image: torch.Tensor, sensitivity: float, min_border_size: int, device: torch.device) -> torch.Tensor:
        """Обрабатывает одно изображение полностью на GPU."""
        img_gpu = (image.to(device) * 255.0).float()
        h, w = img_gpu.shape[:2]
        need_lama = False

        # 1. Пробуем детектировать человека через YOLO (быстрее и точнее)
        person_bbox = self._detect_person_bbox_yolo(img_gpu, device)

        # 2. Fallback на детекцию контента по цвету
        content_bbox = self._find_content_bbox_gpu(img_gpu)

        # Объединяем bbox'ы — берём максимальный охват
        if person_bbox and content_bbox:
            py1, py2, px1, px2 = person_bbox
            cy1, cy2, cx1, cx2 = content_bbox
            final_bbox = (min(py1, cy1), max(py2, cy2), min(px1, cx1), max(px2, cx2))
            logger.info(f"Combined bbox (person + content): {final_bbox}")
        elif person_bbox:
            final_bbox = person_bbox
            logger.info(f"Using person bbox: {final_bbox}")
        else:
            final_bbox = content_bbox
            logger.info(f"Using content bbox: {final_bbox}")

        crop = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}

        # 3. Проверяем края — режем ТОЛЬКО если чисто серый/чёрный/белый И нет контента
        for side in ['left', 'right']:
            is_pure, edge_color = self._is_pure_border_color(img_gpu, side)
            detected_border = self._detect_border_gpu(img_gpu, side, sensitivity, min_border_size)

            if is_pure and detected_border > 0:
                # Дополнительная проверка — есть ли контент на границе обрезки
                has_content = self._check_content_at_edge(img_gpu, side, detected_border)

                if has_content:
                    logger.info(f"{side}: has content at border {detected_border}, not cropping")
                    crop[side] = 0
                elif final_bbox:
                    _, _, bx1, bx2 = final_bbox
                    if side == 'left':
                        safe_limit = max(0, bx1 - self.SAFE_MARGIN_HORIZONTAL)
                    else:
                        safe_limit = max(0, w - bx2 - self.SAFE_MARGIN_HORIZONTAL)

                    # НОВОЕ: Проверяем динамику контента на границе обрезки
                    # Если высокая динамика — игнорируем safe space и режем полностью
                    has_dynamics = self._is_colorful_content_at_border(img_gpu, side, detected_border)

                    if has_dynamics:
                        # Высокая динамика — чёткий край контента, режем полностью
                        crop[side] = detected_border
                        logger.info(f"{side}: high dynamics, ignoring safe space, crop={detected_border}")
                    elif detected_border > safe_limit:
                        # Обрезка хочет зайти в safe space — ищем чёткую границу контента
                        sharp_edge, ambiguous_start = self._find_sharp_content_edge(img_gpu, side, safe_limit, detected_border)
                        if sharp_edge is not None:
                            # Нашли чёткую границу — режем до неё
                            crop[side] = sharp_edge
                            logger.info(f"{side}: found sharp edge at {sharp_edge}, cropping to it (was {detected_border})")
                        elif ambiguous_start is not None:
                            # Есть неоднозначная зона — режем до неё и помечаем для LaMa
                            crop[side] = ambiguous_start
                            need_lama = True
                            logger.info(f"{side}: ambiguous zone at {ambiguous_start}, crop there + LaMa")
                        else:
                            # Нет чёткой границы — останавливаемся на safe limit
                            crop[side] = safe_limit
                            logger.info(f"{side}: no sharp edge in safe zone, crop={safe_limit} (wanted {detected_border})")
                    else:
                        # Обрезка не заходит в safe space — просто режем
                        crop[side] = detected_border
                        logger.info(f"{side}: pure border, crop={crop[side]} (within safe limit)")
                else:
                    crop[side] = detected_border
            else:
                logger.info(f"{side}: not pure border or no border detected")
                crop[side] = 0

        # 4. Анализируем верх/низ
        top_type = self._analyze_edge_strip_gpu(img_gpu, 'top')
        bottom_type = self._analyze_edge_strip_gpu(img_gpu, 'bottom')

        logger.info(f"Edge types: top={top_type}, bottom={bottom_type}")
        logger.info(f"Detected borders: left={crop['left']}, right={crop['right']}")

        top_border = self._detect_border_gpu(img_gpu, 'top', sensitivity, min_border_size)
        bottom_border = self._detect_border_gpu(img_gpu, 'bottom', sensitivity, min_border_size)

        if final_bbox:
            by1, by2, bx1, bx2 = final_bbox
            safe_top = max(0, by1 - self.SAFE_MARGIN_VERTICAL)
            safe_bottom = max(0, h - by2 - self.SAFE_MARGIN_VERTICAL)

            # НОВОЕ: Проверяем динамику контента на границах top/bottom
            top_has_dynamics = self._is_colorful_content_at_border(img_gpu, 'top', top_border) if top_border > 0 else False
            bottom_has_dynamics = self._is_colorful_content_at_border(img_gpu, 'bottom', bottom_border) if bottom_border > 0 else False

            if top_border > 0:
                if top_has_dynamics:
                    # Высокая динамика — игнорируем safe space, режем полностью
                    crop['top'] = top_border
                    logger.info(f"Top: high dynamics, ignoring safe space, crop={top_border}")
                elif top_border > safe_top:
                    # Сначала ищем чёткую границу контента в safe zone
                    sharp_edge, ambiguous_start = self._find_sharp_content_edge(img_gpu, 'top', safe_top, top_border)

                    if sharp_edge is not None:
                        crop['top'] = sharp_edge
                        logger.info(f"Top: found sharp edge at {sharp_edge}, cropping to it")
                    elif ambiguous_start is not None:
                        crop['top'] = ambiguous_start
                        need_lama = True
                        logger.info(f"Top: ambiguous zone at {ambiguous_start}, crop there + LaMa")
                    elif top_type == 'uniform':
                        fill_color = self._get_border_color_gpu(img_gpu, 'top')
                        img_gpu[safe_top:top_border, :, :3] = fill_color
                        crop['top'] = top_border
                        logger.info(f"Top: uniform, filled {safe_top}-{top_border}, crop={top_border}")
                    elif top_type == 'dynamic':
                        crop['top'] = top_border
                        logger.info(f"Top: dynamic edge, full crop={top_border}")
                    else:
                        need_lama = True
                        crop['top'] = safe_top
                        logger.info(f"Top: mixed, no sharp edge, crop to safe={safe_top}, LaMa needed")
                else:
                    crop['top'] = top_border

            if bottom_border > 0:
                if bottom_has_dynamics:
                    # Высокая динамика — игнорируем safe space, режем полностью
                    crop['bottom'] = bottom_border
                    logger.info(f"Bottom: high dynamics, ignoring safe space, crop={bottom_border}")
                elif bottom_border > safe_bottom:
                    sharp_edge, ambiguous_start = self._find_sharp_content_edge(img_gpu, 'bottom', safe_bottom, bottom_border)

                    if sharp_edge is not None:
                        crop['bottom'] = sharp_edge
                        logger.info(f"Bottom: found sharp edge at {sharp_edge}, cropping to it")
                    elif ambiguous_start is not None:
                        crop['bottom'] = ambiguous_start
                        need_lama = True
                        logger.info(f"Bottom: ambiguous zone at {ambiguous_start}, crop there + LaMa")
                    elif bottom_type == 'uniform':
                        fill_color = self._get_border_color_gpu(img_gpu, 'bottom')
                        img_gpu[h-bottom_border:h-safe_bottom, :, :3] = fill_color
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

        if need_lama and final_bbox:
            img_gpu = self._apply_lama_to_borders_gpu(img_gpu, crop, final_bbox, top_type, bottom_type)

        logger.info(f"Final crop: top={crop['top']}, bottom={crop['bottom']}, left={crop['left']}, right={crop['right']}")

        new_h = h - crop['top'] - crop['bottom']
        new_w = w - crop['left'] - crop['right']

        if new_h < h * self.MIN_CONTENT_RATIO or new_w < w * self.MIN_CONTENT_RATIO:
            logger.warning(
                f"Crop too aggressive: {new_h}x{new_w} < {h * self.MIN_CONTENT_RATIO:.0f}x{w * self.MIN_CONTENT_RATIO:.0f}, "
                f"returning original image"
            )
            return (img_gpu / 255.0).clamp(0, 1).cpu()

        y1 = crop['top']
        y2 = h - crop['bottom'] if crop['bottom'] > 0 else h
        x1 = crop['left']
        x2 = w - crop['right'] if crop['right'] > 0 else w

        if y1 >= y2 or x1 >= x2:
            logger.warning(f"Invalid crop bounds: y1={y1}, y2={y2}, x1={x1}, x2={x2}, returning original")
            return (img_gpu / 255.0).clamp(0, 1).cpu()

        result = (img_gpu[y1:y2, x1:x2] / 255.0).clamp(0, 1).cpu()
        return result

    def _get_border_color_gpu(self, img: torch.Tensor, side: str) -> torch.Tensor:
        """Получает РЕАЛЬНЫЙ цвет рамки для заливки на GPU (без нормализации)."""
        h, w = img.shape[:2]
        sample_size = 10

        if side == 'top':
            sample = img[0:sample_size, :, :3]
        else:
            sample = img[h-sample_size:h, :, :3]

        pixels = sample.reshape(-1, 3)
        median = pixels.median(dim=0).values
        # Возвращаем реальный цвет БЕЗ нормализации к чёрному/белому
        return median

    def _apply_lama_to_borders_gpu(self, img_gpu: torch.Tensor, crop: Dict[str, int], content_bbox: Tuple[int, int, int, int], top_type: str, bottom_type: str) -> torch.Tensor:
        """Применяет LaMa для mixed областей."""
        if not HAS_LAMA:
            logger.warning("LaMa not available, skipping border inpainting")
            return img_gpu

        try:
            h, w = img_gpu.shape[:2]
            by1, by2, bx1, bx2 = content_bbox

            mask_gpu = torch.zeros(h, w, device=img_gpu.device, dtype=torch.uint8)

            if top_type == 'mixed':
                top_border = self._detect_border_gpu(img_gpu, 'top', 15, 5)
                safe_top = max(0, by1 - self.SAFE_MARGIN_VERTICAL)
                if top_border > safe_top:
                    mask_gpu[safe_top:top_border, :] = 255

            if bottom_type == 'mixed':
                bottom_border = self._detect_border_gpu(img_gpu, 'bottom', 15, 5)
                safe_bottom = max(0, h - by2 - self.SAFE_MARGIN_VERTICAL)
                if bottom_border > safe_bottom:
                    mask_gpu[h-bottom_border:h-safe_bottom, :] = 255

            if mask_gpu.max() == 0:
                return img_gpu

            logger.info("Applying LaMa for mixed border areas")

            img_np = img_gpu.cpu().numpy().astype(np.uint8)
            mask_np = mask_gpu.cpu().numpy()

            lama = SimpleLama()
            result = lama(Image.fromarray(img_np), Image.fromarray(mask_np))

            return torch.from_numpy(np.array(result)).float().to(img_gpu.device)

        except Exception as e:
            logger.warning(f"LaMa failed: {e}")
            return img_gpu

    def _analyze_edge_strip_gpu(self, img: torch.Tensor, side: str) -> str:
        """Анализирует полосу с края на GPU: uniform, dynamic, mixed."""
        h, w = img.shape[:2]

        if side == 'top':
            strip = img[0:self.EDGE_SIZE, :, :3]
        elif side == 'bottom':
            strip = img[h-self.EDGE_SIZE:h, :, :3]
        elif side == 'left':
            strip = img[:, 0:self.EDGE_SIZE, :3]
        else:
            strip = img[:, w-self.EDGE_SIZE:w, :3]

        pixels = strip.reshape(-1, 3).float()
        std = pixels.std(dim=0).mean().item()
        median = pixels.median(dim=0).values
        saturation = (median.max() - median.min()).item()

        if std < self.UNIFORM_STD_THRESHOLD and saturation < self.SATURATION_THRESHOLD:
            return 'uniform'
        elif std > self.DYNAMIC_STD_THRESHOLD:
            return 'dynamic'
        else:
            return 'mixed'

    def _find_content_bbox_gpu(self, img: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
        """Находит bbox контента на GPU (не чёрное/белое/серое)."""
        h, w = img.shape[:2]

        if img.shape[2] >= 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img[:, :, 0]

        content_mask = ((gray > self.BLACK_THRESHOLD) & (gray < self.WHITE_THRESHOLD)).float()

        if img.shape[2] >= 3:
            max_ch = img[:, :, :3].max(dim=2).values
            min_ch = img[:, :, :3].min(dim=2).values
            saturation = max_ch - min_ch
            color_mask = (saturation > self.COLOR_SATURATION_THRESHOLD).float()
            content_mask = torch.maximum(content_mask, color_mask)

        content_mask = _morphology_open_close_gpu(content_mask, kernel_size=5)

        coords = torch.where(content_mask > 0.5)
        if len(coords[0]) == 0:
            return None

        y1 = coords[0].min().item()
        y2 = coords[0].max().item()
        x1 = coords[1].min().item()
        x2 = coords[1].max().item()

        return (y1, y2, x1, x2)

    def _is_grayscale_color_gpu(self, rgb: torch.Tensor) -> bool:
        """Проверяет, является ли цвет оттенком серого на GPU."""
        saturation = rgb.max() - rgb.min()
        return saturation.item() < self.SATURATION_THRESHOLD

    def _detect_border_gpu(self, img: torch.Tensor, side: str, sensitivity: float, min_size: int) -> int:
        """Двойной проход детекции рамки на GPU."""
        border_pass1 = self._detect_border_by_lines_gpu(img, side, sensitivity, min_size)
        border_pass2 = self._detect_border_by_corner_gpu(img, side, sensitivity, min_size, border_pass1)
        return max(border_pass1, border_pass2)

    def _detect_border_by_lines_gpu(self, img: torch.Tensor, side: str, sensitivity: float, min_size: int) -> int:
        """Проход 1: детекция по полным линиям на GPU."""
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

        sample_lines = min(5, max_scan)
        sample_pixels = torch.cat([get_line(i) for i in range(sample_lines)], dim=0).float()
        ref_color = sample_pixels.median(dim=0).values

        if not self._is_grayscale_color_gpu(ref_color):
            return 0

        border = 0
        for i in range(max_scan):
            line = get_line(i).float()
            line_median = line.median(dim=0).values

            if self._is_grayscale_color_gpu(line_median):
                diff = (line_median - ref_color).abs().mean().item()
                if diff < sensitivity:
                    border = i + 1
                else:
                    break
            else:
                break

        return border if border >= min_size else 0

    def _detect_border_by_corner_gpu(self, img: torch.Tensor, side: str, sensitivity: float, min_size: int, start_offset: int) -> int:
        """Проход 2: проверка угла на GPU."""
        h, w = img.shape[:2]
        corner_size = min(self.MAX_CORNER_SIZE, h // self.CORNER_SIZE_RATIO, w // self.CORNER_SIZE_RATIO)

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

        if corner.numel() == 0 or max_scan <= 0:
            return 0

        ref_color = corner.reshape(-1, 3).float().median(dim=0).values

        if not self._is_grayscale_color_gpu(ref_color):
            return 0

        extra_border = 0
        for i in range(max_scan):
            line = get_line(i).float()
            line_median = line.median(dim=0).values

            if self._is_grayscale_color_gpu(line_median):
                diff = (line_median - ref_color).abs().mean().item()
                if diff < sensitivity * self.SECOND_PASS_SENSITIVITY_MULTIPLIER:
                    extra_border = i + 1
                else:
                    break
            else:
                break

        total = start_offset + extra_border
        return total if extra_border >= min_size else start_offset


class SmartScreenshotCleaner:
    """
    Находит UI элементы через YOLO модели и удаляет их через LaMa инпейнтинг.
    Использует OmniParser для детекции UI элементов.
    """

    _models_cache = {}  # Кэш для всех моделей
    _model_device = None

    # Константы класса
    DEFAULT_CONFIDENCE = 0.1      # Низкий порог для лучшего распознавания UI
    MAX_BOX_RATIO = 0.20          # Максимальный размер UI элемента (20% картинки)
    SURROUNDING_MARGIN = 10
    BOX_EXPAND_PIXELS = 10        # Расширение маски вокруг UI элементов
    YOLO_IMGSZ = 1280             # Разрешение для YOLO
    IOU_THRESHOLD = 0.5           # Порог IoU для дедупликации боксов

    # Зоны UI (доля изображения от края)
    UI_EDGE_ZONE = 0.25           # 25% от края считается UI зоной
    UI_CENTER_SAFE_ZONE = 0.4     # Центральные 40% защищены от детекции текста

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
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > img_area * self.MAX_BOX_RATIO:
                logger.info(f"Skipping large box {x1},{y1}-{x2},{y2}: {box_area/img_area*100:.1f}% of image")
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

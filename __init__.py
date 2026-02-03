from .nodes.border_crop import AutoBorderCrop, SmartScreenshotCleaner, ImageSwitch, MaskSwitch

NODE_CLASS_MAPPINGS = {
    "AutoBorderCrop": AutoBorderCrop,
    "SmartScreenshotCleaner": SmartScreenshotCleaner,
    "ImageSwitch": ImageSwitch,
    "MaskSwitch": MaskSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoBorderCrop": "HappyIn Auto Border Crop",
    "SmartScreenshotCleaner": "HappyIn Screenshot Cleaner",
    "ImageSwitch": "HappyIn Image Switch (If Empty)",
    "MaskSwitch": "HappyIn Mask Switch (If Empty)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

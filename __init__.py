from .nodes.border_crop import AutoBorderCrop, SmartScreenshotCleaner, ImageSwitch, MaskSwitch

NODE_CLASS_MAPPINGS = {
    "HappyIn_AutoBorderCrop": AutoBorderCrop,
    "HappyIn_ScreenshotCleaner": SmartScreenshotCleaner,
    "HappyIn_ImageSwitch": ImageSwitch,
    "HappyIn_MaskSwitch": MaskSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HappyIn_AutoBorderCrop": "HappyIn Auto Border Crop",
    "HappyIn_ScreenshotCleaner": "HappyIn Screenshot Cleaner",
    "HappyIn_ImageSwitch": "HappyIn Image Switch (If Empty)",
    "HappyIn_MaskSwitch": "HappyIn Mask Switch (If Empty)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

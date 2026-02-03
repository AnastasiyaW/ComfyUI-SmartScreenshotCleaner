from .nodes.border_crop import AutoBorderCrop, SmartScreenshotCleaner

NODE_CLASS_MAPPINGS = {
    "AutoBorderCrop": AutoBorderCrop,
    "SmartScreenshotCleaner": SmartScreenshotCleaner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoBorderCrop": "HappyIn Auto Border Crop",
    "SmartScreenshotCleaner": "HappyIn Screenshot Cleaner",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

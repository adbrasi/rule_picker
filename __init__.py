"""Rule34 Picker — Fetch images from Rule34 API with smart caching."""

from .rule34_picker import Rule34Picker

NODE_CLASS_MAPPINGS = {
    "Rule34Picker": Rule34Picker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Rule34Picker": "Rule34 Picker",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

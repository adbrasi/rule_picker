"""Rule34 Picker — Fetch images from Rule34 API with smart caching."""

from .rule34_picker import Rule34Picker
from .tag_cleaner import TagCleaner
from .prompt_comma_fixer import PromptCommaFixer

NODE_CLASS_MAPPINGS = {
    "Rule34Picker": Rule34Picker,
    "TagCleaner": TagCleaner,
    "PromptCommaFixer": PromptCommaFixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Rule34Picker": "Rule34 Picker",
    "TagCleaner": "Tag Cleaner (Blacklist)",
    "PromptCommaFixer": "Prompt Comma Fixer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

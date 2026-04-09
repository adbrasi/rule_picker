"""Node for removing duplicate tags from a comma-separated prompt."""
from __future__ import annotations


class TagDedup:
    """Removes duplicate tags from a comma-separated prompt string.

    Keeps the first occurrence of each tag, preserves order and comma+space
    formatting. Comparison is case-insensitive and ignores extra whitespace.
    """

    CATEGORY = "text/rule34"
    FUNCTION = "dedup_tags"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("deduped_prompt",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Removes duplicate tags from a comma-separated prompt. "
        "Keeps first occurrence, preserves order and formatting."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Comma-separated prompt string to deduplicate.",
                }),
            },
        }

    def dedup_tags(self, prompt: str) -> tuple:
        if not prompt.strip():
            return (prompt,)

        parts = prompt.split(",")
        seen = set()
        unique = []

        for part in parts:
            tag = part.strip()
            key = tag.lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(tag)

        return (", ".join(unique),)

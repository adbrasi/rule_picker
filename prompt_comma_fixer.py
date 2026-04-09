"""Node for fixing comma spacing in prompts."""
from __future__ import annotations

import re


class PromptCommaFixer:
    """Fixes comma formatting in prompts so every comma is followed by a space.

    Converts 'item1,item2' to 'item1, item2'.
    Also normalizes multiple spaces and trims whitespace.
    """

    CATEGORY = "text/rule34"
    FUNCTION = "fix_commas"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fixed_prompt",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Fixes comma spacing in prompts. Ensures every comma is followed "
        "by exactly one space: 'tag1,tag2' becomes 'tag1, tag2'."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The prompt string to fix.",
                }),
            },
        }

    def fix_commas(self, prompt: str) -> tuple:
        # Replace comma followed by non-space (or no space) with comma+space
        result = re.sub(r",\s*", ", ", prompt)

        # Clean up multiple spaces
        result = " ".join(result.split())

        return (result.strip(),)

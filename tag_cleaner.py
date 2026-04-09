"""Node for removing unwanted tags from a tag string."""
from __future__ import annotations


class TagCleaner:
    """Removes blacklisted tags or substrings from an input tag string.

    First tries to match and remove whole tags (space-separated tokens).
    If a blacklist entry is not found as a whole tag, removes it as a
    substring from the entire string.
    """

    CATEGORY = "text/rule34"
    FUNCTION = "clean_tags"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_tags",)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Removes unwanted tags or substrings from a tag string. "
        "Enter one blacklist entry per line."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The input tag string to clean.",
                }),
                "blacklist": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One entry per line. Whole tags are removed first; if not found as a tag, removed as substring.",
                }),
            },
        }

    def clean_tags(self, tags: str, blacklist: str) -> tuple:
        if not blacklist.strip():
            return (tags,)

        # Parse blacklist entries (one per line, trimmed, skip empty)
        entries = [e.strip() for e in blacklist.splitlines() if e.strip()]

        # Split tags into tokens
        tokens = tags.split()
        result_tokens = list(tokens)

        remaining_entries = []

        for entry in entries:
            entry_lower = entry.lower()
            # Try to remove as whole tag (case-insensitive)
            found = False
            new_tokens = []
            for token in result_tokens:
                if token.lower() == entry_lower:
                    found = True
                else:
                    new_tokens.append(token)
            if found:
                result_tokens = new_tokens
            else:
                remaining_entries.append(entry)

        # Rebuild string from remaining tokens
        result = " ".join(result_tokens)

        # For entries not found as whole tags, remove as substring
        for entry in remaining_entries:
            result = result.replace(entry, "")

        # Clean up multiple spaces and trim
        result = " ".join(result.split())

        return (result,)

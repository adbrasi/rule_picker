"""Rule34 API client with pagination, retries, and rate limiting."""

import json
import time
import urllib.parse
import urllib.request
from typing import Any

BASE_URL = "https://api.rule34.xxx/index.php"

# Extensions we can load as images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

# Maximum posts the API returns per request
API_LIMIT = 1000

# Delay between paginated requests (seconds)
PAGE_DELAY = 0.5

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5


def _is_image_url(url: str) -> bool:
    """Check if a URL points to an image file."""
    if not url:
        return False
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def _build_url(tags: str, sort_tag: str, limit: int, pid: int,
               api_key: str, user_id: str) -> str:
    """Build the API request URL."""
    combined_tags = tags.strip()
    if sort_tag:
        combined_tags = f"{combined_tags} {sort_tag}".strip()

    params = {
        "page": "dapi",
        "s": "post",
        "q": "index",
        "json": "1",
        "limit": min(limit, API_LIMIT),
        "pid": pid,
        "tags": combined_tags,
    }
    if api_key:
        params["api_key"] = api_key
    if user_id:
        params["user_id"] = user_id

    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def _fetch_page(url: str, timeout: int = 30) -> list[dict[str, Any]]:
    """Fetch a single page from the API with retries."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ComfyUI-Rule34-Picker/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw or raw.strip() == b"":
                    return []
                data = json.loads(raw)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    if data.get("success") is False:
                        msg = data.get("message", "unknown error")
                        raise RuntimeError(f"Rule34 API error: {msg}")
                    return []
                return []
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))

    raise RuntimeError(
        f"Rule34 API request failed after {MAX_RETRIES} retries: {last_error}"
    )


def fetch_posts(
    tags: str,
    sort_tag: str = "",
    max_pages: int = 2,
    api_key: str = "",
    user_id: str = "",
) -> list[dict[str, Any]]:
    """Fetch posts from Rule34 API with pagination.

    Returns a list of post dicts with only image posts (videos filtered out).
    Each post dict contains: id, file_url, sample_url, score, tags, rating.
    """
    all_posts: list[dict[str, Any]] = []

    for page in range(max_pages):
        url = _build_url(tags, sort_tag, API_LIMIT, page, api_key, user_id)
        raw_posts = _fetch_page(url)

        if not raw_posts:
            break

        for p in raw_posts:
            file_url = p.get("file_url", "")
            sample_url = p.get("sample_url", "")

            # Skip non-image posts (videos, etc.)
            if not _is_image_url(file_url) and not _is_image_url(sample_url):
                continue

            all_posts.append({
                "id": p["id"],
                "file_url": file_url,
                "sample_url": sample_url,
                "score": p.get("score", 0),
                "tags": p.get("tags", ""),
                "rating": p.get("rating", ""),
            })

        # If we got less than the limit, there are no more pages
        if len(raw_posts) < API_LIMIT:
            break

        # Rate limit between pages
        if page < max_pages - 1:
            time.sleep(PAGE_DELAY)

    return all_posts

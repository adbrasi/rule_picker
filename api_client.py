"""Rule34 API client with pagination, retries, and rate limiting."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

BASE_URL = "https://api.rule34.xxx/index.php"

# File type groups
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".gif", ".webm", ".mp4", ".avi", ".mov", ".mkv", ".flv"}
ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Maximum posts the API returns per request
API_LIMIT = 1000

# Delay between paginated requests (seconds)
PAGE_DELAY = 0.5

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
RETRY_BACKOFF = 1.5

# Max API JSON response size (50 MB)
MAX_API_RESPONSE_BYTES = 50 * 1024 * 1024


def _get_url_extension(url: str) -> str:
    """Extract the file extension from a URL."""
    if not url:
        return ""
    path = urllib.parse.urlparse(url).path.lower()
    for ext in ALL_MEDIA_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ""


def _matches_media_filter(url: str, media_filter: str) -> bool:
    """Check if a URL matches the given media filter.

    media_filter: "images", "videos", or "all"
    """
    ext = _get_url_extension(url)
    if not ext:
        return False
    if media_filter == "images":
        return ext in IMAGE_EXTENSIONS
    if media_filter == "videos":
        return ext in VIDEO_EXTENSIONS
    return ext in ALL_MEDIA_EXTENSIONS


def _redact_url(url: str) -> str:
    """Remove sensitive query params from URL for logging."""
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    for key in ("api_key", "user_id"):
        if key in qs:
            qs[key] = ["[REDACTED]"]
    safe_query = urllib.parse.urlencode(qs, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=safe_query))


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


def _fetch_page(url: str, timeout: int = DEFAULT_TIMEOUT,
                max_retries: int = DEFAULT_MAX_RETRIES) -> list[dict[str, Any]]:
    """Fetch a single page from the API with retries."""
    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Rule34-Picker/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(MAX_API_RESPONSE_BYTES + 1)
                if len(raw) > MAX_API_RESPONSE_BYTES:
                    raise RuntimeError("API response exceeds size limit")
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
            if attempt < max_retries - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))

    raise RuntimeError(
        f"Rule34 API request failed after {max_retries} retries "
        f"(url={_redact_url(url)}): {last_error}"
    )


def fetch_posts(
    tags: str,
    sort_tag: str = "",
    max_pages: int = 2,
    api_key: str = "",
    user_id: str = "",
    media_filter: str = "images",
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict[str, Any]]:
    """Fetch posts from Rule34 API with pagination.

    Returns a list of post dicts filtered by media_filter ("images", "videos", "all").
    Each post dict contains: id, file_url, sample_url, score, tags, rating.
    """
    all_posts: list[dict[str, Any]] = []

    for page in range(max_pages):
        url = _build_url(tags, sort_tag, API_LIMIT, page, api_key, user_id)
        raw_posts = _fetch_page(url, timeout=timeout, max_retries=max_retries)

        if not raw_posts:
            break

        for p in raw_posts:
            post_id = p.get("id")
            if post_id is None:
                continue

            file_url = p.get("file_url", "")
            sample_url = p.get("sample_url", "")

            # Filter by media type
            primary_url = file_url or sample_url
            if not _matches_media_filter(primary_url, media_filter):
                continue

            all_posts.append({
                "id": post_id,
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

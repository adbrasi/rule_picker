"""Node for fetching images from Rule34 API."""
from __future__ import annotations

import hashlib
import io
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

from . import api_client
from . import cache_manager

SORT_OPTIONS = [
    "score:desc",
    "score:asc",
    "id:desc",
    "id:asc",
    "updated:desc",
    "updated:asc",
]

MEDIA_TYPE_OPTIONS = [
    "images",
    "videos",
    "all",
]

# Max parallel download threads
MAX_DOWNLOAD_WORKERS = 8

# Hard wall-clock timeout per download (seconds)
HARD_DOWNLOAD_TIMEOUT = 120

# Allowed URL hosts for downloads
_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_HOSTS = frozenset({
    "rule34.xxx",
    "api.rule34.xxx",
    "us.rule34.xxx",
    "img3.rule34.xxx",
    "wimg.rule34.xxx",
})


def _validate_url(url: str) -> None:
    """Validate URL scheme and host to prevent SSRF."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme}")
    host = parsed.netloc.split(":")[0].lower()
    if not any(host == h or host.endswith("." + h) for h in _ALLOWED_HOSTS):
        raise ValueError(f"Disallowed host: {parsed.netloc}")


def _download_file(url: str, timeout: int, max_size_mb: int) -> bytes:
    """Download a file from a URL with size limit."""
    _validate_url(url)
    max_bytes = max_size_mb * 1024 * 1024
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Rule34-Picker/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"File too large (>{max_size_mb} MB)")
    return data


def _bytes_to_image_tensor(data: bytes) -> torch.Tensor:
    """Convert raw image bytes to ComfyUI IMAGE tensor (1, H, W, C) float32."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.width == 0 or img.height == 0:
        raise ValueError("Decoded image has zero dimensions")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class Rule34Picker:
    """Fetches media from Rule34 API with caching and batch support.

    Uses a disk-based cache to store post metadata from the first API call.
    Subsequent executions consume from the cache, advancing a cursor.
    Seed changes trigger re-execution (use fixed to pause, random/increment to advance).
    """

    CATEGORY = "image/rule34"
    FUNCTION = "pick_images"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Fetches images from Rule34 API with disk-based caching and parallel downloads. "
        "Change the seed to trigger a new batch. Use refresh_cache to re-fetch from API."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Tags to search for. Supports Rule34 syntax: tag1 tag2 -excluded rating:safe user:username",
                }),
                "sort_by": (SORT_OPTIONS, {"default": "score:desc"}),
                "media_type": (MEDIA_TYPE_OPTIONS, {
                    "default": "images",
                    "tooltip": "Filter by media type: images (png/jpg/webp), videos (gif/webm/mp4), or all.",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of images to fetch per execution.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Change this value to trigger re-execution and fetch a new batch. Use fixed to pause, random/increment to advance.",
                }),
                "never_repeat": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip already-seen images. Resets silently when all images have been shown.",
                }),
                "max_pages": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of API pages to fetch (up to 1000 posts per page).",
                }),
                "use_full_resolution": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use full resolution (file_url) instead of sample (sample_url). Slower but higher quality.",
                }),
                "refresh_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force re-fetch from API, ignoring existing cache.",
                }),
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 300,
                    "step": 5,
                    "tooltip": "Socket timeout in seconds for each download/API request.",
                }),
                "max_retries": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Maximum number of retries for failed API requests.",
                }),
                "max_file_size_mb": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 500,
                    "step": 10,
                    "tooltip": "Maximum file size in MB for each downloaded image/video.",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Rule34 API key (optional, increases rate limits).",
                }),
                "user_id": ("STRING", {
                    "default": "",
                    "tooltip": "Rule34 user ID (required if api_key is set).",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, tags, sort_by, media_type, batch_size, seed,
                   never_repeat, max_pages, use_full_resolution, refresh_cache,
                   timeout, max_retries, max_file_size_mb,
                   api_key="", user_id=""):
        """Hash all meaningful inputs so any change triggers re-execution."""
        key = (
            f"{tags}|{sort_by}|{media_type}|{batch_size}|{seed}|{never_repeat}"
            f"|{max_pages}|{use_full_resolution}|{refresh_cache}"
        )
        return hashlib.md5(key.encode()).hexdigest()

    def pick_images(
        self,
        tags: str,
        sort_by: str,
        media_type: str,
        batch_size: int,
        seed: int,
        never_repeat: bool,
        max_pages: int,
        use_full_resolution: bool,
        refresh_cache: bool,
        timeout: int,
        max_retries: int,
        max_file_size_mb: int,
        api_key: str = "",
        user_id: str = "",
    ) -> tuple:
        tags = tags.strip()
        if not tags:
            raise ValueError("Tags field cannot be empty.")

        sort_tag = f"sort:{sort_by}"

        # Check cache or fetch from API
        if refresh_cache:
            cache_manager.invalidate_cache(tags, sort_by)

        cache = cache_manager.load_cache(tags, sort_by)
        if cache is None:
            print(f"[Rule34Picker] Fetching posts for tags='{tags}' sort='{sort_tag}' media={media_type} pages={max_pages}")
            posts = api_client.fetch_posts(
                tags=tags,
                sort_tag=sort_tag,
                max_pages=max_pages,
                api_key=api_key,
                user_id=user_id,
                media_filter=media_type,
                timeout=timeout,
                max_retries=max_retries,
            )
            if not posts:
                raise RuntimeError(
                    f"No posts found for tags: {tags} (media_type={media_type})"
                )
            cache_manager.save_cache(tags, sort_by, posts)
            print(f"[Rule34Picker] Cached {len(posts)} posts")

        # Get next batch from cache
        batch_posts = cache_manager.get_next_batch(
            tags, sort_by, batch_size, never_repeat
        )
        if not batch_posts:
            raise RuntimeError("No posts available. Try refreshing the cache.")

        # Download images in parallel
        url_key = "file_url" if use_full_resolution else "sample_url"
        fallback_key = "sample_url" if use_full_resolution else "file_url"

        def _download_post(post: dict) -> tuple:
            url = post.get(url_key) or post.get(fallback_key)
            if not url:
                return post["id"], None
            try:
                data = _download_file(url, timeout=timeout, max_size_mb=max_file_size_mb)
                tensor = _bytes_to_image_tensor(data)
                return post["id"], tensor
            except Exception as exc:
                print(f"[Rule34Picker] Failed to download post {post['id']}: {exc}")
                return post["id"], None

        workers = min(batch_size, MAX_DOWNLOAD_WORKERS)
        results: dict = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_post, p): p for p in batch_posts}
            for future in as_completed(futures, timeout=HARD_DOWNLOAD_TIMEOUT * len(futures)):
                try:
                    post_id, tensor = future.result(timeout=HARD_DOWNLOAD_TIMEOUT)
                    if tensor is not None:
                        results[post_id] = tensor
                except Exception as exc:
                    post = futures[future]
                    print(f"[Rule34Picker] Download failed/timed out for post {post['id']}: {exc}")

        # Maintain original order
        images: list = []
        for post in batch_posts:
            if post["id"] in results:
                images.append(results[post["id"]])

        if not images:
            raise RuntimeError(
                "Failed to download any images from the current batch. "
                "Check your network connection or try different tags."
            )

        ids = [p["id"] for p in batch_posts if p["id"] in results]
        print(f"[Rule34Picker] Delivered {len(images)} images (IDs: {ids})")

        return (images,)

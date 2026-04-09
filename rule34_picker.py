"""ComfyUI node for fetching images from Rule34 API."""

import io
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

# Max parallel download threads
MAX_DOWNLOAD_WORKERS = 8

# Download timeout per image (seconds)
DOWNLOAD_TIMEOUT = 30


def _download_image(url: str) -> Image.Image:
    """Download an image from a URL and return as PIL Image."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ComfyUI-Rule34-Picker/1.0"},
    )
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor (1, H, W, C) float32."""
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class Rule34Picker:
    """Fetches images from Rule34 API with caching and batch support.

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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "tag1 tag2 -excluded_tag rating:safe",
                }),
                "sort_by": (SORT_OPTIONS, {"default": "score:desc"}),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "never_repeat": ("BOOLEAN", {"default": True}),
                "max_pages": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of API pages to fetch (1000 posts per page)",
                }),
                "use_full_resolution": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use full resolution (file_url) instead of sample (sample_url). Slower but higher quality.",
                }),
                "refresh_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force re-fetch from API, ignoring existing cache.",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Rule34 API key (optional, increases rate limits)",
                }),
                "user_id": ("STRING", {
                    "default": "",
                    "tooltip": "Rule34 user ID (required if api_key is set)",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        """Seed controls re-execution. Fixed seed = cached result, changing seed = re-run."""
        return seed

    def pick_images(
        self,
        tags: str,
        sort_by: str,
        batch_size: int,
        seed: int,
        never_repeat: bool,
        max_pages: int,
        use_full_resolution: bool,
        refresh_cache: bool,
        api_key: str = "",
        user_id: str = "",
    ) -> tuple[list[torch.Tensor]]:
        tags = tags.strip()
        if not tags:
            raise ValueError("Tags field cannot be empty.")

        sort_tag = f"sort:{sort_by}"

        # Check cache or fetch from API
        if refresh_cache:
            cache_manager.invalidate_cache(tags, sort_by)

        cache = cache_manager.load_cache(tags, sort_by)
        if cache is None:
            print(f"[Rule34Picker] Fetching posts for tags='{tags}' sort='{sort_tag}' pages={max_pages}")
            posts = api_client.fetch_posts(
                tags=tags,
                sort_tag=sort_tag,
                max_pages=max_pages,
                api_key=api_key,
                user_id=user_id,
            )
            if not posts:
                raise RuntimeError(
                    f"No image posts found for tags: {tags}"
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

        def _download_post(post: dict) -> tuple[int, torch.Tensor | None]:
            url = post.get(url_key) or post.get(fallback_key)
            if not url:
                return post["id"], None
            try:
                img = _download_image(url)
                return post["id"], _image_to_tensor(img)
            except Exception as exc:
                print(f"[Rule34Picker] Failed to download post {post['id']}: {exc}")
                return post["id"], None

        workers = min(batch_size, MAX_DOWNLOAD_WORKERS)
        results: dict[int, torch.Tensor] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_post, p): p for p in batch_posts}
            for future in as_completed(futures):
                post_id, tensor = future.result()
                if tensor is not None:
                    results[post_id] = tensor

        # Maintain original order
        images: list[torch.Tensor] = []
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

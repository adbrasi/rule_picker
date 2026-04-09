"""Cache manager for Rule34 post metadata.

Stores fetched post IDs and metadata on disk so subsequent node executions
can consume from the cache without re-fetching the API. Tracks a cursor
position and a set of seen post IDs for never-repeat mode.
"""

import hashlib
import json
import os
import time
from typing import Any, Optional

CACHE_DIR = os.path.join("/tmp", "rule34_cache")


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(tags: str, sort: str) -> str:
    """Deterministic cache key from tags + sort."""
    raw = f"{tags.strip().lower()}|{sort}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def load_cache(tags: str, sort: str) -> Optional[dict[str, Any]]:
    """Load cache from disk. Returns None if not found."""
    _ensure_cache_dir()
    path = _cache_path(_cache_key(tags, sort))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(tags: str, sort: str, posts: list[dict[str, Any]]) -> dict[str, Any]:
    """Save a fresh cache to disk. Resets cursor and seen list."""
    _ensure_cache_dir()
    cache_data = {
        "tags": tags,
        "sort": sort,
        "posts": posts,
        "cursor": 0,
        "seen": [],
        "fetched_at": time.time(),
        "total_posts": len(posts),
    }
    path = _cache_path(_cache_key(tags, sort))
    with open(path, "w") as f:
        json.dump(cache_data, f)
    return cache_data


def _write_cache(tags: str, sort: str, cache_data: dict[str, Any]) -> None:
    """Persist updated cache data to disk."""
    _ensure_cache_dir()
    path = _cache_path(_cache_key(tags, sort))
    with open(path, "w") as f:
        json.dump(cache_data, f)


def invalidate_cache(tags: str, sort: str) -> None:
    """Delete the cache file for the given tags+sort."""
    _ensure_cache_dir()
    path = _cache_path(_cache_key(tags, sort))
    if os.path.exists(path):
        os.remove(path)


def get_next_batch(
    tags: str,
    sort: str,
    batch_size: int,
    never_repeat: bool,
) -> list[dict[str, Any]]:
    """Get the next batch of posts from the cache.

    Advances the cursor. If never_repeat is True, skips already-seen posts
    and resets silently when all posts have been seen.

    Returns a list of post dicts (may be shorter than batch_size if not
    enough posts exist).
    """
    cache = load_cache(tags, sort)
    if cache is None:
        return []

    posts = cache["posts"]
    total = len(posts)
    if total == 0:
        return []

    cursor = cache.get("cursor", 0)
    seen = set(cache.get("seen", []))
    batch: list[dict[str, Any]] = []

    if never_repeat:
        # Collect unseen posts starting from cursor
        checked = 0
        i = cursor
        while len(batch) < batch_size and checked < total:
            post = posts[i % total]
            if post["id"] not in seen:
                batch.append(post)
                seen.add(post["id"])
            i += 1
            checked += 1

        # Exhausted all posts — silent reset and continue filling
        if len(batch) < batch_size:
            seen.clear()
            # Re-add what we just picked in this batch
            for p in batch:
                seen.add(p["id"])
            i = 0
            checked = 0
            while len(batch) < batch_size and checked < total:
                post = posts[i % total]
                if post["id"] not in seen:
                    batch.append(post)
                    seen.add(post["id"])
                i += 1
                checked += 1

        cache["cursor"] = i % total if total > 0 else 0
        cache["seen"] = list(seen)
    else:
        # Simple sequential access, wraps around
        for j in range(batch_size):
            idx = (cursor + j) % total
            batch.append(posts[idx])
        cache["cursor"] = (cursor + batch_size) % total

    _write_cache(tags, sort, cache)
    return batch

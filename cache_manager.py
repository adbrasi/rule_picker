"""Cache manager for Rule34 post metadata.

Stores fetched post IDs and metadata on disk so subsequent node executions
can consume from the cache without re-fetching the API. Tracks a cursor
position and a set of seen post IDs for never-repeat mode.

Uses atomic writes (write to temp file + os.replace) to prevent corruption
on crash or concurrent access.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from typing import Any, Optional

CACHE_DIR = os.path.join(tempfile.gettempdir(), "rule34_cache")

# Per-key locks to prevent concurrent read-modify-write races
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    """Get or create a per-key lock."""
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(tags: str, sort: str) -> str:
    """Deterministic cache key from tags + sort.

    Normalizes whitespace so 'cat  dog' and 'cat dog' hit the same cache.
    """
    normalized_tags = " ".join(tags.strip().lower().split())
    raw = f"{normalized_tags}|{sort}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _atomic_write(path: str, data: dict[str, Any]) -> None:
    """Write JSON data atomically using temp file + os.replace."""
    _ensure_cache_dir()
    dir_ = os.path.dirname(path)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        raise RuntimeError(
            f"[Rule34Picker] Failed to write cache (disk full?): {e}"
        ) from e


def load_cache(tags: str, sort: str) -> Optional[dict[str, Any]]:
    """Load cache from disk. Returns None if not found or corrupted."""
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
    key = _cache_key(tags, sort)
    cache_data = {
        "tags": tags,
        "sort": sort,
        "posts": posts,
        "cursor": 0,
        "seen": [],
        "fetched_at": time.time(),
        "total_posts": len(posts),
    }
    lock = _get_lock(key)
    with lock:
        _atomic_write(_cache_path(key), cache_data)
    return cache_data


def invalidate_cache(tags: str, sort: str) -> None:
    """Delete the cache file for the given tags+sort."""
    key = _cache_key(tags, sort)
    path = _cache_path(key)
    lock = _get_lock(key)
    with lock:
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
    key = _cache_key(tags, sort)
    lock = _get_lock(key)

    with lock:
        cache = load_cache(tags, sort)
        if cache is None:
            return []

        posts = cache["posts"]
        total = len(posts)
        if total == 0:
            return []

        # Cap effective batch to total posts to avoid infinite loops
        effective_batch = min(batch_size, total)

        cursor = cache.get("cursor", 0)
        seen = set(cache.get("seen", []))
        batch: list[dict[str, Any]] = []

        if never_repeat:
            # First pass: collect unseen posts starting from cursor
            i = cursor
            checked = 0
            while len(batch) < effective_batch and checked < total:
                post = posts[i % total]
                if post["id"] not in seen:
                    batch.append(post)
                    seen.add(post["id"])
                i += 1
                checked += 1

            # Exhausted all posts — silent reset and continue filling
            if len(batch) < effective_batch:
                seen.clear()
                for p in batch:
                    seen.add(p["id"])
                remaining = effective_batch - len(batch)
                i = 0
                added = 0
                while added < remaining and i < total:
                    post = posts[i]
                    if post["id"] not in seen:
                        batch.append(post)
                        seen.add(post["id"])
                        added += 1
                    i += 1

            cache["cursor"] = i % total if total > 0 else 0
            cache["seen"] = list(seen)
        else:
            # Simple sequential access, wraps around
            for j in range(effective_batch):
                idx = (cursor + j) % total
                batch.append(posts[idx])
            cache["cursor"] = (cursor + effective_batch) % total

        _atomic_write(_cache_path(key), cache)
        return batch

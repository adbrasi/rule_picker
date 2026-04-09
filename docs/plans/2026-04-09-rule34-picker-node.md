# Rule34 Picker ComfyUI Node - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a ComfyUI custom node that fetches images from Rule34 API with intelligent caching, parallel downloads, and seed-based triggering.

**Architecture:** Single package with 3 modules: API client (handles Rule34 communication with retries), cache manager (persists post IDs to /tmp, tracks cursor and seen posts), and main node (ComfyUI integration with parallel image downloads). Cache is keyed by tags+sort, stores post metadata on first fetch, subsequent runs consume from cache advancing a cursor.

**Tech Stack:** Python 3.x, torch, PIL/Pillow, urllib (all already available in ComfyUI), concurrent.futures for parallel downloads.

---

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Cache location | `/tmp/rule34_cache/` | Cleans on reboot, no persistence needed |
| Cache key | SHA256(tags+sort)[:16] | Deterministic, collision-resistant |
| Image download | ThreadPoolExecutor(max_workers=8) | Max parallelism without overwhelming |
| Output format | `OUTPUT_IS_LIST = (True,)` | Handles different image sizes, max compatibility |
| Video filtering | Skip non-image URLs at cache time | Avoid wasting bandwidth |
| Sort integration | Append `sort:<field>:<order>` to tags param | API native support |
| Trigger mechanism | `IS_CHANGED` returns seed value | Fixed=no rerun, random/increment=rerun |

---

### Task 1: API Client Module

**Files:**
- Create: `api_client.py`

Handles all Rule34 API communication: post fetching with pagination, retries, rate limiting.

### Task 2: Cache Manager Module

**Files:**
- Create: `cache_manager.py`

Manages post ID cache on disk. Tracks cursor position, seen posts for never-repeat mode. Handles cache invalidation and silent reset on exhaustion.

### Task 3: Main Node

**Files:**
- Create: `rule34_picker.py`

ComfyUI node class with all inputs (tags, sort, batch_size, seed, never_repeat, max_pages, api_key, user_id, use_full_res, refresh_cache). Parallel image download, PIL->tensor conversion, list output.

### Task 4: Package Init

**Files:**
- Create: `__init__.py`

Node registration with NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

### Task 5: Verification

Launch 3 sonnet agents:
1. Bug & logic review
2. Edge case & error handling review
3. ComfyUI compatibility review

"""Utility functions for dataset maintenance and inspection usable by both CLI and Gradio UIs.

These helpers avoid duplicating logic that previously lived only inside *gradio_app.py* so that
`menu.py` can import them and offer equivalent features in the interactive CLI.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError

from .dataset import SUPPORTED_IMAGE_EXTS, INPUT_DIR  # type: ignore
from .utils.common import resolve_dataset_path  # Use shared function


__all__ = [
    "resolve_dataset_path",
    "dataset_stats",
    "scan_duplicates",
    "delete_all_images",
]


###############################################################################
# Helper to resolve dataset folder (input/ or external source)
###############################################################################

# resolve_dataset_path is now imported from utils.common


###############################################################################
# Statistics / preview helpers
###############################################################################

def _parse_captions_tokens(caption: str) -> List[str]:
    STOP_TOKENS = {
        "a",
        "the",
        "and",
        "of",
        "in",
        "on",
        "with",
        "is",
        "to",
        "for",
        "at",
        "by",
        "from",
        "as",
        "an",
        "her",
        "his",
        "she",
        "he",
        "it",
        "its",
    }
    return [tok.lower() for tok in caption.split() if len(tok) >= 4 and tok.lower() not in STOP_TOKENS]


def dataset_stats(name: str, *, sample_size: int = 24) -> dict:
    """Compute quick stats of *dataset*.

    Returns a dict with keys:
        images: total number of image files in folder
        txt:     number of .txt caption files
        avg_w, avg_h: average resolution of up to *sample_size* images
        top_tokens:   list[(token,count)] of most common words in captions
    """

    folder = resolve_dataset_path(name)
    if folder is None or not folder.exists():
        raise FileNotFoundError(f"Dataset not found: {name}")

    # gather list of images (limit *sample_size* for perf)
    paths: List[Path] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        paths.extend(folder.glob(f"*.{ext}"))
    paths = sorted(paths)[:sample_size]

    # Counters
    captions_tokens: List[str] = []
    sizes: List[Tuple[int, int]] = []

    for p in paths:
        # resolution
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            pass
        # caption tokens
        txt = p.with_suffix(".txt")
        if txt.exists():
            caption = txt.read_text("utf-8", errors="ignore").strip()
            captions_tokens.extend(_parse_captions_tokens(caption))

    # global counts
    total_imgs = sum(len(list(folder.glob(f"*.{ext}"))) for ext in SUPPORTED_IMAGE_EXTS)
    total_txt = len(list(folder.glob("*.txt")))

    avg_w = int(sum(s[0] for s in sizes) / len(sizes)) if sizes else 0
    avg_h = int(sum(s[1] for s in sizes) / len(sizes)) if sizes else 0
    common_tokens = Counter(captions_tokens).most_common(10)

    return {
        "images": total_imgs,
        "txt": total_txt,
        "avg_w": avg_w,
        "avg_h": avg_h,
        "top_tokens": common_tokens,
    }


###############################################################################
# Duplicate / corrupt scan
###############################################################################

def scan_duplicates(name: str) -> Tuple[int, int]:
    """Return (#duplicates, #corrupt) inside *dataset* using md5 hash and PIL.verify()"""

    folder = resolve_dataset_path(name)
    if folder is None or not folder.exists():
        raise FileNotFoundError(f"Dataset not found: {name}")

    hashes = {}
    dup_count = 0
    corrupt_count = 0

    img_paths = []
    for ext in SUPPORTED_IMAGE_EXTS:
        img_paths.extend(folder.glob(f"*.{ext}"))

    for p in img_paths:
        try:
            with Image.open(p) as im:
                im.verify()
        except (UnidentifiedImageError, OSError):
            corrupt_count += 1
            continue

        try:
            h = hashlib.md5(p.read_bytes()).hexdigest()
        except Exception:
            continue

        if h in hashes:
            dup_count += 1
        else:
            hashes[h] = p

    return dup_count, corrupt_count


###############################################################################
# Delete images helper
###############################################################################

def delete_all_images(name: str) -> int:
    """Delete every supported image in dataset and return number removed."""

    folder = resolve_dataset_path(name)
    if folder is None or not folder.exists():
        raise FileNotFoundError(f"Dataset not found: {name}")

    count = 0
    for ext in SUPPORTED_IMAGE_EXTS:
        for p in folder.glob(f"*.{ext}"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
    return count 
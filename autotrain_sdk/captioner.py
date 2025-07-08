from __future__ import annotations

"""Caption generation utilities using Microsoft Florence-2.

This module provides two public functions:
    - ``caption_image``: caption a single image file.
    - ``caption_dataset``: iterate over all images inside ``input/<dataset>`` and write
      ``.txt`` files next to them if missing (or when *overwrite* is ``True``).

The first call loads the model & processor lazily and keeps them in memory so we
can caption many images without paying the load cost each time.

Example
-------
>>> from autotrain_sdk.captioner import caption_dataset
>>> n = caption_dataset("b09g13")
>>> print(f"{n} captions created")
"""

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .paths import INPUT_DIR, OUTPUT_DIR
from .dataset import SUPPORTED_IMAGE_EXTS


__all__ = [
    "caption_image",
    "caption_dataset",
    "rename_and_caption_dataset",
]

# ---------------------------------------------------------------------------
# Internal helpers / lazy loading
# ---------------------------------------------------------------------------

_MODEL_ID = "microsoft/Florence-2-base"
_MODEL: AutoModelForCausalLM | None = None
_PROC: AutoProcessor | None = None

def _ensure_loaded(device: str) -> tuple[AutoProcessor, AutoModelForCausalLM]:
    """Load model & processor (once) and return them."""

    global _MODEL, _PROC

    if _MODEL is None:
        _PROC = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        _MODEL = (
            AutoModelForCausalLM
            .from_pretrained(_MODEL_ID, trust_remote_code=True)
            .to(device)
            .eval()
        )
    # mypy: we know they are not None here
    return _PROC, _MODEL  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def caption_image(
    img_path: str | Path,
    *,
    device: str | None = None,
    max_new_tokens: int = 128,
) -> str:
    """Return a caption for *img_path*.

    Parameters
    ----------
    img_path: str | Path
        Path to the image file.
    device: str | None, default None
        ``"cuda"``/``"cpu"`` or explicit GPU id.  Auto-detect when *None*.
    max_new_tokens: int
        Generation length cap.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = _ensure_loaded(device)

    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    img = Image.open(img_path).convert("RGB")
    inputs = processor(text="<MORE_DETAILED_CAPTION>", images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
            early_stopping=False,
        )

    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        raw,
        task="<MORE_DETAILED_CAPTION>",
        image_size=img.size,
    )
    return result["<MORE_DETAILED_CAPTION>"]


def caption_dataset(
    name: str,
    *,
    device: str | None = None,
    max_new_tokens: int = 128,
    overwrite: bool = False,
) -> int:
    """Generate captions for every image in *input/<name>*.

    Returns the number of captions written.
    """

    folder = INPUT_DIR / name
    if not folder.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found inside {INPUT_DIR}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    written = 0
    for path in folder.iterdir():
        if path.suffix.lower().lstrip(".") not in SUPPORTED_IMAGE_EXTS:
            continue
        txt_path = path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            continue
        caption = caption_image(path, device=device, max_new_tokens=max_new_tokens)
        txt_path.write_text(caption + "\n", encoding="utf-8")
        written += 1
    return written


def rename_and_caption_dataset(
    dataset: str,
    trigger: str,
    *,
    device: str | None = None,
    max_new_tokens: int = 128,
    overwrite: bool = True,
) -> int:
    """Rename images inside *input/<dataset>* to ``{trigger}_<n>.ext`` and generate captions.

    Captions are saved to ``{trigger}_<n>.txt`` with the format::
        {trigger}, <generated caption>

    Parameters
    ----------
    dataset: str
        Folder name inside *input/*.
    trigger: str
        Base name / keyword.
    overwrite: bool, default True
        If ``False``, skip caption file when it already exists.

    Returns
    -------
    int
        Number of captions created.
    """

    folder = INPUT_DIR / dataset
    if not folder.exists():
        raise FileNotFoundError(f"Dataset '{dataset}' not found in {INPUT_DIR}")

    # Filter image files
    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower().lstrip(".") in SUPPORTED_IMAGE_EXTS]
    if not images:
        return 0

    pad = max(2, len(str(len(images))))  # at least 2 digits

    created = 0
    for idx, old_path in enumerate(images, 1):
        ext = old_path.suffix.lower()
        new_name = f"{trigger}_{idx:0{pad}}{ext}"
        new_path = folder / new_name

        # If file exists and is the same, keep; else rename
        if old_path != new_path:
            # Handle potential conflict by skipping renaming if target exists
            if new_path.exists():
                # Simple fallback: skip renaming; proceed with captioning
                pass
            else:
                old_path.rename(new_path)
        else:
            new_path = old_path  # unchanged

        # Caption path
        txt_path = new_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            continue

        cap = caption_image(new_path, device=device, max_new_tokens=max_new_tokens)
        txt_path.write_text(f"{trigger}, {cap}\n", encoding="utf-8")

        # NOTE: Copying to output structure is handled after captioning by higher-level helpers
        created += 1

    return created 
from __future__ import annotations
"""TOML preset configurator (Phase 1).

This module provides generic utilities to load, modify, and save
TOML files while maintaining the **same structure** required by sd-scripts.

In the future (Phase 2+), this will be replaced by Pydantic models; for now, we maintain a
minimum viable product that allows the CLI/Gradio to edit arbitrary values.
"""

from pathlib import Path
from typing import Any, Dict
import importlib
import toml
import datetime, shutil

from .paths import BATCH_CONFIG_DIR, get_project_root

__all__ = [
    "load_config",
    "save_config",
    "update_config",
    "generate_presets",
    "generate_presets_for_dataset",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a TOML file and return it as a ``dict``.

    No validation is performed: all keys are preserved.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return toml.load(path)


def save_config(path: str | Path, config: Dict[str, Any]) -> None:
    """Save a ``dict`` in TOML format"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ts = path.stat().st_mtime
        timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(f".{timestamp}.bak")
        shutil.copy2(path, backup_path)
    with path.open("w", encoding="utf-8") as f:
        toml.dump(config, f)


def update_config(path: str | Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a TOML file in-place and return it.

    Only keys present in *updates* are overwritten; the rest are
    kept intact.
    """

    cfg = load_config(path)
    cfg.update(updates)
    save_config(path, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Preset generation wrapper (reusing combined_script.py)
# ---------------------------------------------------------------------------

def generate_presets() -> None:
    """Invokes functions from ``combined_script.py`` to regenerate presets.

    It's equivalent to running the script itself but without extra
    side effects (prints are ignored). This allows the CLI/Gradio to refresh
    TOML files if the user requests it.
    """

    project_root = str(get_project_root())
    module = importlib.import_module("combined_script")

    # Each function returns a bool
    for fn_name in (
        "process_flux_checkpoint",
        "process_flux_lora",
        "process_sdxl_nude",
    ):
        fn = getattr(module, fn_name, None)
        if fn is None:
            raise AttributeError(f"combined_script.{fn_name} not found")
        fn(project_root)

    # We don't return anything: TOMLs are written to BatchConfig/


# Convenient alias for future CLI
refresh_presets = generate_presets


def generate_presets_for_dataset(dataset_name: str) -> list[Path]:
    """Generates presets and returns the list of .toml files for *dataset_name*."""

    # Snapshot existing files before
    pre_existing: dict[str, set[str]] = {}
    for sub in ("Flux", "FluxLORA", "Nude"):
        dir_path = BATCH_CONFIG_DIR / sub
        pre_existing[sub] = {p.name for p in dir_path.glob("*.toml")}

    generate_presets()

    bc_root = BATCH_CONFIG_DIR
    result: list[Path] = []
    for sub in ("Flux", "FluxLORA", "Nude"):
        dir_path = bc_root / sub
        target = dir_path / f"{dataset_name}.toml"
        if target.exists():
            result.append(target)

        # Delete new files that do not correspond to the dataset
        for p in dir_path.glob("*.toml"):
            if p.name != f"{dataset_name}.toml" and p.name not in pre_existing[sub]:
                p.unlink(missing_ok=True)

    return result 
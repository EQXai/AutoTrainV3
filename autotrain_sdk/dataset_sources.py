from __future__ import annotations
"""Manage additional dataset source directories.

The user can register one or more *source folders* that contain datasets
(each dataset = sub-directory with images + captions).  These sources are
scanned to list datasets that are not already present under ``input/`` and can
optionally be imported automatically before training.

Data is persisted in ``.gradio/dataset_sources.json``.
"""

from pathlib import Path
import json
from typing import Dict, List

from .paths import get_project_root

SOURCES_FILE: Path = get_project_root() / ".gradio" / "dataset_sources.json"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _read_raw() -> List[str]:
    try:
        if SOURCES_FILE.exists():
            return json.loads(SOURCES_FILE.read_text())  # type: ignore[return-value]
    except Exception:
        pass
    return []


def load_sources() -> List[Path]:
    """Return list of valid source directories."""

    paths = []
    for p_str in _read_raw():
        p = Path(p_str).expanduser().resolve()
        if p.is_dir():
            paths.append(p)
    return paths


def save_sources(paths: List[str | Path]):
    """Overwrite the persisted list of sources with *paths*."""

    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    norm = [str(Path(p).expanduser().resolve()) for p in paths if Path(p).expanduser().is_dir()]
    SOURCES_FILE.write_text(json.dumps(norm, indent=2))


def add_source(path: str | Path) -> List[Path]:
    """Add a new source directory (if valid) and return updated list."""

    p = Path(path).expanduser().resolve()
    paths = load_sources()
    if p.is_dir() and p not in paths:
        paths.append(p)
        save_sources(paths)
    return paths


def remove_source(path: str | Path) -> List[Path]:
    p = Path(path).expanduser().resolve()
    paths = load_sources()
    paths = [pp for pp in paths if pp != p]
    save_sources(paths)
    return paths

# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def list_external_datasets() -> Dict[str, Path]:
    """Return mapping {dataset_name: absolute_path} from all sources."""

    mapping: Dict[str, Path] = {}
    for src in load_sources():
        for d in src.iterdir():
            if d.is_dir():
                mapping.setdefault(d.name, d)  # first occurrence wins
    return mapping


def find_dataset_path(name: str) -> Path | None:
    return list_external_datasets().get(name) 
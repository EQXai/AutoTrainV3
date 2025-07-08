from __future__ import annotations

"""Sweep / grid-search utilities (Sprint 1).

This module generates multiple training presets (TOML files) for the *same*
 dataset by expanding a parameter grid.  Each variant gets its own
 `output_dir` and `logging_dir` so runs never overwrite each other.

The public helpers are:

• parse_grid(param_flags)   → dict[str, list]
• expand_grid(grid)         → list[dict]
• make_variant_name(opts)   → str
• generate_variant(dataset, profile, overrides) → Path

The CLI sub-command will rely on these helpers (see *cli.py*).
"""

from pathlib import Path
from itertools import product
from typing import Dict, List, Any
import re, datetime, toml

from .paths import BATCH_CONFIG_DIR, OUTPUT_DIR
from .configurator import load_config, save_config

__all__ = [
    "parse_grid",
    "expand_grid",
    "make_variant_name",
    "generate_variant",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_value(v: str) -> Any:
    """Convert CLI strings to python types (int, float, bool) when possible."""

    l = v.lower()
    if l in {"true", "false"}:
        return l == "true"
    # int / float
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v  # keep as str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_grid(param_flags: List[str]) -> Dict[str, List[Any]]:
    """Parse a list of ``key=v1,v2`` strings coming from CLI flags.

    Returns a dict mapping key → list[values]. Raises ``ValueError`` on bad
    input.
    """

    grid: Dict[str, List[Any]] = {}
    for flag in param_flags:
        if "=" not in flag:
            raise ValueError(f"Bad syntax '{flag}'. Use key=v1,v2")
        key, raw_vals = flag.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in '{flag}'")
        vals = [s.strip() for s in re.split(r",|;", raw_vals) if s.strip()]
        if not vals:
            raise ValueError(f"No values given for '{key}'")
        grid[key] = [_coerce_value(v) for v in vals]
    return grid


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of the *grid* dict.

    Example::
        {"lr": [1e-5, 4e-6], "epochs": [1,3]}
        → [{"lr": 1e-5, "epochs": 1}, {"lr": 1e-5, "epochs": 3}, ...]
    """

    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = []
    for vals in product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, vals)))
    return combos


def make_variant_name(overrides: Dict[str, Any]) -> str:
    """Return a compact deterministic variant identifier.

    Keys are sorted alphabetically; special chars replaced by nothing.
    """

    parts = []
    for k in sorted(overrides):
        v = overrides[k]
        safe_v = str(v).replace("/", "").replace(" ", "")
        parts.append(f"{k}{safe_v}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Preset generation
# ---------------------------------------------------------------------------

def generate_variant(dataset: str, profile: str, overrides: Dict[str, Any]) -> Path:
    """Create a variant TOML file and return its path.

    Parameters
    ----------
    dataset : str
        e.g. ``b09g13`` (without extension).
    profile : str
        "Flux", "FluxLORA" or "Nude".
    overrides : dict
        Keys / values that override the base preset.
    """

    profile = profile.strip()
    base_path = BATCH_CONFIG_DIR / profile / f"{dataset}.toml"
    if not base_path.is_file():
        raise FileNotFoundError(f"Base preset not found: {base_path}")

    cfg = load_config(base_path)

    # Apply overrides -------------------------------------------------------
    cfg.update(overrides)

    variant_name = make_variant_name(overrides) or "base"

    # Adjust output / logging dirs so runs are isolated ---------------------
    now_ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = OUTPUT_DIR / dataset / variant_name / profile / now_ts
    model_dir = run_root / "model"
    log_dir = run_root / "log"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg["output_dir"] = str(model_dir.as_posix())
    cfg["logging_dir"] = str(log_dir.as_posix())

    # Some templates use 'wandb_run_name'
    cfg.setdefault("wandb_run_name", variant_name)

    # ----------------------------------------------------------------------
    variant_path = BATCH_CONFIG_DIR / profile / f"{dataset}__{variant_name}.toml"
    save_config(variant_path, cfg)

    return variant_path 
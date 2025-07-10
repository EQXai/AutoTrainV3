from __future__ import annotations

from pathlib import Path

# Calculate the project root once, assuming this file lives in autotrain_sdk/paths.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ------------------------------
# Path helper functions
# ------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the AutoTrainV2 project root."""

    return PROJECT_ROOT


# Convenience constants (Path objects)
INPUT_DIR: Path = PROJECT_ROOT / "input"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
BATCH_CONFIG_DIR: Path = PROJECT_ROOT / "BatchConfig"
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
SD_SCRIPTS_DIR: Path = PROJECT_ROOT / "sd-scripts"
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"

# Add any other frequently-used directories here as needed. 

# ---------------------------------------------------------------------------
# Helper to compute run directory depending on remote config
# ---------------------------------------------------------------------------


def compute_run_dir(dataset: str, profile: str, *, timestamp: str | None = None, job_id: str | None = None) -> Path:
    """Return the directory where training outputs should be stored.

    If AUTO_REMOTE_BASE is set and AUTO_REMOTE_DIRECT=="1", the path will be
    inside that base directory.  Otherwise it falls back to output/<dataset>/<profile>/<timestamp>.
    The folder name is ``{dataset}_{profile}_{job_id or ts}`` when remote direct.
    """

    import os, datetime, uuid

    ts = timestamp or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jid = job_id or uuid.uuid4().hex[:8]

    remote_base = os.getenv("AUTO_REMOTE_BASE")
    direct_env = os.getenv("AUTO_REMOTE_DIRECT")
    if remote_base is None or direct_env is None:
        try:
            import json
            # Prefer project-level file, fallback to home dir
            candidate_paths = [PROJECT_ROOT / ".gradio" / "integrations.json", Path.home() / ".gradio" / "integrations.json"]
            for cfg_path in candidate_paths:
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text())
                    remote_base = remote_base or cfg.get("AUTO_REMOTE_BASE")
                    direct_env = direct_env or cfg.get("AUTO_REMOTE_DIRECT")
                    # stop after first match
                    break
        except Exception:
            # Gracefully ignore any issues reading cfg
            pass

    direct = (direct_env == "1")

    if remote_base and direct:
        name = f"{dataset}_{profile}_{jid}"
        return Path(remote_base) / name

    # local fallback path inside project
    return OUTPUT_DIR / dataset / profile / ts 
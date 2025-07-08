from __future__ import annotations
"""Persistent registry of training runs for Model Organizer.

Records are stored as JSON objects in one file ``.gradio/runs.json`` next to other
metadata so that both the Gradio UI and CLI can access them quickly.

The registry is append-only but updates an existing job *id* in-place when the
same `job_id` is written again (for example when status changes from *pending*
→ *running* → *done*).
"""

from pathlib import Path
import json
import threading
import datetime as _dt
from typing import Any, Dict

from .paths import get_project_root

_REG_PATH = get_project_root() / ".gradio" / "runs.json"
_LOCK = threading.Lock()


def _load() -> Dict[str, dict]:
    if _REG_PATH.exists():
        try:
            return json.loads(_REG_PATH.read_text())
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    _REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(_REG_PATH)


def upsert(job, extra: dict[str, Any] | None = None) -> None:  # type: ignore[valid-type]
    """Insert or update *job* entry.

    Parameters
    ----------
    job
        Instance of ``Job`` (comes from ``job_manager``).
    extra
        Arbitrary extra key-values to merge/override.
    """
    with _LOCK:
        data = _load()
        rec = data.get(job.id, {})

        # core fields always updated
        rec.update({
            "job_id": job.id,
            "dataset": job.dataset,
            "profile": job.profile,
            "status": job.status.value if hasattr(job.status, "value") else str(job.status),
            "run_dir": str(job.run_dir),
            "toml_path": str(job.toml_path),
            "gpu_ids": job.gpu_ids,
            "experiment_id": job.experiment_id,
        })

        if extra:
            rec.update(extra)

        # --- Auto-extract details from TOML & metrics if available ---
        try:
            import toml, json as _json, hashlib, os

            # Parse training config TOML once (cheap)
            if job.toml_path and Path(job.toml_path).exists():
                try:
                    cfg = toml.load(job.toml_path)
                    rec.setdefault("epochs", cfg.get("epochs") or cfg.get("max_train_epochs"))
                    rec.setdefault("batch_size", cfg.get("train_batch_size") or cfg.get("batch_size"))
                    rec.setdefault("learning_rate", cfg.get("learning_rate") or cfg.get("lr"))
                    rec.setdefault("resolution", cfg.get("resolution"))
                    rec.setdefault("network_dim", cfg.get("network_dim"))
                    rec.setdefault("network_alpha", cfg.get("network_alpha"))
                except Exception:
                    pass

            # Parse metrics.json if present
            metrics_path = Path(rec["run_dir"]) / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics = _json.loads(metrics_path.read_text())
                    rec.update(metrics)
                except Exception:
                    pass

            # Compute model hash for first .safetensors file
            if "model_hash" not in rec:
                try:
                    for p in Path(rec["run_dir"]).glob("*.safetensors"):
                        h = hashlib.sha256()
                        with p.open("rb") as fp:
                            while chunk := fp.read(8192):
                                h.update(chunk)
                        rec["model_hash"] = h.hexdigest()[:16]
                        break
                except Exception:
                    pass
        except Exception:
            pass

        data[job.id] = rec
        _save(data)


def ensure_initial_scan() -> None:
    """Populate registry from existing output/ runs if file absent."""
    if _REG_PATH.exists():
        return

    from .paths import OUTPUT_DIR
    from glob import glob
    import toml, os

    runs: dict[str, dict] = {}

    for ds_dir in OUTPUT_DIR.glob("*"):
        for mode_dir in ds_dir.glob("*"):
            for run_dir in mode_dir.glob("*"):
                if not run_dir.is_dir():
                    continue
                job_id = run_dir.name
                rec: dict[str, Any] = {
                    "job_id": job_id,
                    "dataset": ds_dir.name,
                    "profile": mode_dir.name,
                    "run_dir": str(run_dir),
                    "status": "unknown",
                }
                toml_path = next(run_dir.glob("*.toml"), None)
                if toml_path:
                    rec["toml_path"] = str(toml_path)
                    try:
                        cfg = toml.load(toml_path)
                        rec["epochs"] = cfg.get("epochs") or cfg.get("max_train_epochs")
                    except Exception:
                        pass
                runs[job_id] = rec

    _save(runs)


# ---------------------------------------------------------------------------
# Rebuild registry (full scan)
# ---------------------------------------------------------------------------


def rebuild(progress_cb: callable | None = None) -> int:
    """Recreate `runs.json` from scratch by scanning `output/`.

    Returns number of records written.
    """
    from .paths import OUTPUT_DIR
    import toml, json as _json, hashlib

    records: dict[str, dict] = {}

    def _hash_first_model(run: Path):
        for p in run.glob("*.safetensors"):
            try:
                h = hashlib.sha256()
                with p.open("rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
                return h.hexdigest()[:16]
            except Exception:
                return None
        return None

    runs_dirs = list(OUTPUT_DIR.rglob("*/"))
    total = len(runs_dirs)
    count = 0
    for rd in runs_dirs:
        if not rd.is_dir():
            continue
        parts = rd.relative_to(OUTPUT_DIR).parts
        if len(parts) != 3:
            continue
        ds, profile, jid = parts
        rec: dict[str, Any] = {
            "job_id": jid,
            "dataset": ds,
            "profile": profile,
            "run_dir": str(rd),
        }

        toml_path = next(rd.glob("*.toml"), None)
        if toml_path:
            rec["toml_path"] = str(toml_path)
            try:
                cfg = toml.load(toml_path)
                rec.update({
                    "epochs": cfg.get("epochs") or cfg.get("max_train_epochs"),
                    "batch_size": cfg.get("train_batch_size") or cfg.get("batch_size"),
                    "learning_rate": cfg.get("learning_rate") or cfg.get("lr"),
                    "resolution": cfg.get("resolution"),
                    "network_dim": cfg.get("network_dim"),
                })
            except Exception:
                pass

        metrics_path = rd / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = _json.loads(metrics_path.read_text())
                rec.update(metrics)
            except Exception:
                pass

        hash_mdl = _hash_first_model(rd)
        if hash_mdl:
            rec["model_hash"] = hash_mdl

        records[jid] = rec
        count += 1
        if progress_cb:
            progress_cb(count, total)

    _save(records)
    return count 
from __future__ import annotations
"""Experiments helper – Fase 0.

Allow planning multiple training variants for the same dataset.
This core module handles the metadata & enqueuing; UI will come later.
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict

from .paths import get_project_root, BATCH_CONFIG_DIR, compute_run_dir
from .configurator import load_config, save_config
from .job_manager import Job, JOB_MANAGER

EXP_DIR = get_project_root() / ".gradio" / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

class RunSpec(Dict):
    """Dictionary with at least profile and overrides keys."""

class Experiment:
    def __init__(self, dataset: str, runs: List[RunSpec]):
        self.id = str(uuid.uuid4())[:8]
        self.dataset = dataset
        self.runs = runs  # list of RunSpec
        self.status = "planned"

    # --- persistence ---
    def path(self) -> Path:
        return EXP_DIR / f"{self.id}.json"

    def save(self):
        data = {"id": self.id, "dataset": self.dataset, "runs": self.runs, "status": self.status}
        self.path().write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(exp_id: str) -> "Experiment":
        p = EXP_DIR / f"{exp_id}.json"
        data = json.loads(p.read_text())
        exp = Experiment(data["dataset"], data["runs"])
        exp.id = data["id"]
        exp.status = data.get("status", "planned")
        return exp

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def create_experiment(dataset: str, variants: List[RunSpec]) -> Experiment:
    """Create experiment metadata and enqueue all jobs."""

    exp = Experiment(dataset, variants)
    exp.save()

    for idx, var in enumerate(variants):
        profile = var.get("profile", "Flux")
        overrides = var.get("overrides", {})

        # base toml path
        toml_base = BATCH_CONFIG_DIR / (profile if profile != "Flux" else "Flux") / f"{dataset}.toml"
        cfg = load_config(toml_base)
        cfg.update(overrides)

        # build unique run dir
        run_dir = compute_run_dir(dataset, profile)
        run_dir.mkdir(parents=True, exist_ok=True)

        # patch config with new paths
        cfg["output_dir"] = str(run_dir)
        cfg["logging_dir"] = str(run_dir / "log")

        patched_path = EXP_DIR / f"tmp_{exp.id}_{idx}.toml"
        save_config(patched_path, cfg)

        job = Job(dataset, profile, patched_path, run_dir, experiment_id=exp.id)
        JOB_MANAGER.enqueue(job)

    exp.status = "running"
    exp.save()
    return exp 
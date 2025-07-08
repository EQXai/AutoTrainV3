from __future__ import annotations

"""Launcher de entrenamiento para sd-scripts (Fase 1).

Construye los comandos `accelerate launch …` de forma programática.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Iterable, List
import logging
from pydantic import ValidationError

from .paths import SD_SCRIPTS_DIR, LOGS_DIR, get_project_root
from .config_models import load_config_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

__all__ = [
    "build_accelerate_command",
    "run_training",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROFILE_BIN = {
    "Nude": "sdxl_train.py",
    "FluxLORA": "flux_train_network.py",
    "Flux": "flux_train.py",
}

DEFAULT_ARGS = {
    "Nude": [
        "--max_grad_norm=0.0",
        "--no_half_vae",
        "--train_text_encoder",
        "--learning_rate_te2=0",
    ],
    "FluxLORA": [],
    "Flux": [],
}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def build_accelerate_command(
    config_file: Path,
    profile: str,
    *,
    venv_python: Path | None = None,
    output_dir: Path | None = None,
    gpu_ids: str | None = None,
) -> List[str]:
    """Devuelve la lista de argumentos para ``subprocess``.

    Parameters
    ----------
    config_file: Path
        Ruta al TOML.
    profile: str
        Uno de ``Nude``, ``FluxLORA`` o ``Flux``.
    venv_python: Path | None
        Python executable a usar. Por defecto el *python* activo.
    """

    profile = profile.strip()
    if profile not in PROFILE_BIN:
        raise ValueError(f"Unknown profile: {profile}")

    script_name = PROFILE_BIN[profile]
    script_path = SD_SCRIPTS_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    python_exe = str(venv_python or sys.executable)

    cmd: List[str] = [
        python_exe,
        "-m",
        "accelerate.commands.launch",
        "--dynamo_backend",
        "no",
        "--dynamo_mode",
        "default",
        "--mixed_precision",
        "bf16",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--num_cpu_threads_per_process",
        "4",
    ]

    # GPU flags solo para Flux/FluxLORA
    if profile != "Nude" and gpu_ids is not None:
        cmd += ["--gpu_ids", gpu_ids]

    cmd += [str(script_path)]

    # if script supports output dir we'll add first so later options override config
    if output_dir is not None:
        cmd += ["--output_dir", str(output_dir)]

    cmd += ["--config_file", str(config_file)]

    cmd += DEFAULT_ARGS.get(profile, [])

    return cmd


def run_training(config_file: Path, profile: str, *, stream: bool = True, gpu_ids: str | None = None) -> int:
    """Ejecuta un entrenamiento y redirige log a `logs/<name>.log`.

    Returns el *returncode* del proceso.
    """

    config_file = Path(config_file)
    profile = profile.strip()

    if not config_file.is_file():
        raise FileNotFoundError(config_file)

    # Validate TOML against TrainingConfig
    try:
        load_config_file(config_file)
    except ValidationError as e:
        logger.error("Config validation failed for '%s':\n%s", config_file.name, e)
        return 1

    file_name = config_file.stem
    log_file = LOGS_DIR / f"{file_name}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = build_accelerate_command(config_file, profile, gpu_ids=gpu_ids)
    logger.info("Running: %s", " ".join(cmd))

    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        if stream:
            for line in proc.stdout:  # type: ignore[attr-defined]
                sys.stdout.write(line)
                f.write(line)
        else:
            proc.communicate()
            if proc.stdout:
                f.write(proc.stdout)

    return proc.wait() 
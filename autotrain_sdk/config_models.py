from __future__ import annotations
"""Typed models for training configuration (Phase 2).

Actualmente los presets TOML se tratan como ``dict`` libres. Este módulo
introduce una representación *typed* usando Pydantic v2 para aportar:

• validación temprana de campos obligatorios
• autocompletado en IDEs / type-checking con MyPy
• conversión automática de rutas ``Path`` ↔︎ ``str`` al serializar

Sólo se declaran las claves comunes a todos los perfiles; el resto se admite
vía ``extra = "allow"`` para no romper compatibilidad. Se podrán crear
subclases específicas (``FluxConfig``, ``FluxLoraConfig``…) que añadan campos
obligatorios adicionales.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict
import toml

__all__ = [
    "TrainingConfig",
    "load_config_file",
    "dump_config_file",
]


class TrainingConfig(BaseModel):
    """Modelo base para presets de *sd-scripts*.

    Sólo un subconjunto de campos se tipa explícitamente; el resto se acepta
    tal cual para mantener flexibilidad.
    """

    # Campos que suelen cambiar dinámicamente
    output_dir: Path = Field(..., description="Ruta donde se guarda el modelo entrenado")
    logging_dir: Path = Field(..., description="Ruta de los logs de entrenamiento")
    train_data_dir: Path = Field(..., description="Carpeta con las imágenes/txt de entrenamiento")
    output_name: str = Field(..., description="Nombre base para checkpoints y logs")

    # Ejemplo de campo opcional presente en algunos perfiles
    sample_prompts: Optional[Path] = None

    # Permitir campos arbitrarios (mantener compatibilidad)
    model_config = ConfigDict(extra="allow")

    # ------------------------------------------------------
    # Validadores / post-procesado
    # ------------------------------------------------------

    @field_validator("output_dir", "logging_dir", "train_data_dir", mode="before")
    @classmethod
    def _expanduser(cls, v):  # type: ignore[return-type]
        """Expande ~ y convierte a Path (si venía como str)."""
        return Path(v).expanduser() if isinstance(v, (str, Path)) else v

    # ------------------------------------------------------
    # Serialización utilitaria
    # ------------------------------------------------------

    def to_toml_dict(self) -> Dict[str, Any]:
        """Dict apto para `toml.dump` conservando tipo *str* en rutas."""

        return {
            k: str(v.as_posix()) if isinstance(v, Path) else v
            for k, v in self.model_dump(exclude_none=True).items()
        }


# ---------------------------------------------------------------------------
# Helper functions (I/O)
# ---------------------------------------------------------------------------


def load_config_file(path: Path | str) -> TrainingConfig:
    """Lee TOML → ``TrainingConfig``."""

    data = toml.load(Path(path))
    return TrainingConfig(**data)


def dump_config_file(cfg: TrainingConfig, path: Path | str) -> None:
    """Serializa ``TrainingConfig`` a TOML."""

    with Path(path).open("w", encoding="utf-8") as f:
        toml.dump(cfg.to_toml_dict(), f) 
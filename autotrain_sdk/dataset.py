from __future__ import annotations

"""Dataset utilities (Phase 1).

Este módulo reemplaza los scripts Bash:
    - 1.Input_Batch_Images.sh
    - 1.2.Output_Batch_Create.sh
    - 1.3.Delete_Input_Output.sh
con funciones Python reutilizables.

Todas las rutas se extraen de ``autotrain_sdk.paths`` para mantener coherencia.
"""

from pathlib import Path
import shutil
import logging
from typing import Iterable, List

from .paths import INPUT_DIR, OUTPUT_DIR, BATCH_CONFIG_DIR, get_project_root

__all__ = [
    "SUPPORTED_IMAGE_EXTS",
    "SUPPORTED_TEXT_EXTS",
    "create_input_folders",
    "populate_output_structure",
    "clean_workspace",
    "populate_output_structure_single",
    "create_sample_prompts",
    "create_sample_prompts_for_dataset",
]

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTS: tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "gif")
SUPPORTED_TEXT_EXTS: tuple[str, ...] = ("txt",)


# ---------------------------------------------------------------------------
# Funciones utilitarias
# ---------------------------------------------------------------------------

def _has_images(folder: Path) -> bool:
    """Return ``True`` if the folder contains at least one supported image file."""

    for ext in SUPPORTED_IMAGE_EXTS:
        if any(folder.glob(f"*.{ext}")):
            return True
    return False


def _copy_files(src: Path, dst: Path) -> int:
    """Copia imágenes y archivos de texto de *src* a *dst* si no existen.

    Devuelve el número de archivos copiados.
    """

    copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for ext in (*SUPPORTED_IMAGE_EXTS, *SUPPORTED_TEXT_EXTS):
        for f in src.glob(f"*.{ext}"):
            target = dst / f.name
            if not target.exists():
                shutil.copy2(f, target)
                copied += 1
    return copied


def _read_base_prompt() -> str:
    """Lee el archivo base_prompt.txt y devuelve su contenido."""
    base_prompt_file = get_project_root() / "txt" / "base_prompt.txt"
    
    if not base_prompt_file.exists():
        logger.warning(f"Base prompt file not found: {base_prompt_file}")
        return "A high-quality photo"
    
    try:
        with base_prompt_file.open("r", encoding="utf-8") as f:
            content = f.read().strip()
        logger.info(f"Base prompt loaded from: {base_prompt_file}")
        return content
    except Exception as e:
        logger.error(f"Failed to read base prompt file: {e}")
        return "A high-quality photo"


def _create_sample_prompts_file(dataset_name: str, output_path: Path) -> bool:
    """Crea el archivo sample_prompts.txt para un dataset específico."""
    try:
        base_prompt = _read_base_prompt()
        prompts_file = output_path / "sample_prompts.txt"
        prompt_content = f"{dataset_name}, {base_prompt}"
        
        with prompts_file.open("w", encoding="utf-8") as f:
            f.write(prompt_content)
        
        logger.info(f"Sample prompts file created: {prompts_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create sample prompts file for '{dataset_name}': {e}")
        return False


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def create_input_folders(names: Iterable[str], *, input_dir: Path | None = None) -> List[Path]:
    """Create subfolders in the main input directory.

    Parameters
    ----------
    names: Iterable[str]
        List of names (whitespace will be stripped). E.g. ["foo", "bar"].
    input_dir: Path | None
        Root directory (defaults to ``INPUT_DIR``).

    Returns
    --------
    List[Path]
        List of created or existing folders.
    """

    root = input_dir or INPUT_DIR
    root.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        folder = root / name
        folder.mkdir(exist_ok=True)
        created.append(folder)
        logger.info("Folder '%s' ready.", folder.relative_to(root.parent))
    return created


def populate_output_structure(*, input_dir: Path | None = None, output_dir: Path | None = None, min_images: int = 0, repeats: int = 30) -> None:
    """Replicates the logic of *1.2.Output_Batch_Create.sh*.

    - Validates that each input folder contains images
    - Creates the ``output/<name>/{img,log,model}`` structure
    - Copies images to ``output/<name>/img/30_<name> person/``
    - Creates sample_prompts.txt files for each dataset

    Parameters
    ----------
    input_dir: Path | None
        Input folder (defaults to ``INPUT_DIR``).
    output_dir: Path | None
        Output folder (defaults to ``OUTPUT_DIR``).
    min_images: int
        If >0, raises ``ValueError`` when a folder doesn't meet the minimum.
    repeats: int
        Number of repeats for the special folder (default 30).
    """

    inp = input_dir or INPUT_DIR
    out = output_dir or OUTPUT_DIR

    if not inp.exists():
        raise FileNotFoundError(f"Input directory '{inp}' does not exist.")

    out.mkdir(parents=True, exist_ok=True)

    valid_folders: list[Path] = []
    for sub in inp.iterdir():
        if sub.is_dir():
            if not _has_images(sub):
                logger.warning("Input folder '%s' contains no images.", sub.name)
                if min_images:
                    raise ValueError(f"Folder '{sub.name}' must contain at least {min_images} images.")
            valid_folders.append(sub)

    for folder in valid_folders:
        name = folder.name
        dest_root = out / name
        (dest_root / "model").mkdir(parents=True, exist_ok=True)
        (dest_root / "log").mkdir(exist_ok=True)
        (dest_root / "img").mkdir(exist_ok=True)

        special = dest_root / "img" / f"{repeats}_{name} person"
        special.mkdir(parents=True, exist_ok=True)

        copied = _copy_files(folder, special)
        logger.info("%d image(s) copied for '%s'.", copied, name)
        
        # Create sample_prompts.txt file
        _create_sample_prompts_file(name, dest_root)

    logger.info("Output structure populated successfully.")


def clean_workspace(*, delete_input: bool = True, delete_output: bool = True, delete_batchconfig: bool = True) -> None:
    """Safely delete workspace folders.

    Similar to *1.3.Delete_Input_Output.sh* but non-interactive.
    """

    def _remove(path: Path):
        if path.exists():
            shutil.rmtree(path)
            logger.info("Deleted '%s'.", path.name)
        else:
            logger.debug("Path '%s' does not exist – skipping.", path.name)

    if delete_input:
        _remove(INPUT_DIR)
    if delete_output:
        _remove(OUTPUT_DIR)
    if delete_batchconfig:
        _remove(BATCH_CONFIG_DIR)


# ------------------------------------------------------------
# Single-dataset variant (used from Gradio interface)
# ------------------------------------------------------------

def populate_output_structure_single(dataset_name: str, *, repeats: int = 30) -> None:
    """Creates/updates the output structure for a single dataset.

    Equivalent to `populate_output_structure` but restricted to one folder.
    It also creates the corresponding sample_prompts.txt file.
    """

    folder = INPUT_DIR / dataset_name
    if not folder.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' does not exist in {INPUT_DIR}")

    out_root = OUTPUT_DIR / dataset_name
    (out_root / "model").mkdir(parents=True, exist_ok=True)
    (out_root / "log").mkdir(exist_ok=True)
    (out_root / "img").mkdir(exist_ok=True)

    special = out_root / "img" / f"{repeats}_{dataset_name} person"
    special.mkdir(parents=True, exist_ok=True)

    copied = _copy_files(folder, special)
    logger.info("%d file(s) copied for '%s'.", copied, dataset_name)

    # Create sample_prompts.txt file
    _create_sample_prompts_file(dataset_name, out_root)

    return None


# ------------------------------------------------------------
# Functions specific to sample_prompts
# ------------------------------------------------------------

def create_sample_prompts() -> int:
    """Creates sample_prompts.txt files for all datasets in output/.
    
    This function replicates the functionality of create_sample_prompts.py which
    was in PorEliminar/.
    
    Returns
    -------
    int
        Number of created/updated sample_prompts.txt files.
    """
    
    if not OUTPUT_DIR.exists():
        logger.error(f"Output directory not found: {OUTPUT_DIR}")
        logger.info("Please run output structure creation first.")
        return 0
    
    # Obtener lista de datasets
    try:
        datasets = [f.name for f in OUTPUT_DIR.iterdir() if f.is_dir()]
    except Exception as e:
        logger.error(f"Failed to list output directory: {e}")
        return 0
    
    if not datasets:
        logger.info(f"No datasets found in: {OUTPUT_DIR}")
        return 0
    
    created_count = 0
    
    for dataset_name in datasets:
        dataset_path = OUTPUT_DIR / dataset_name
        if _create_sample_prompts_file(dataset_name, dataset_path):
            created_count += 1
    
    logger.info(f"Created/updated {created_count} sample_prompts.txt files")
    return created_count


def create_sample_prompts_for_dataset(dataset_name: str) -> bool:
    """Creates a sample_prompts.txt file for a specific dataset.
    
    Parameters
    ----------
    dataset_name: str
        Name of the dataset in output/.
        
    Returns
    -------
    bool
        True if created successfully, False otherwise.
    """
    
    dataset_path = OUTPUT_DIR / dataset_name
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False
    
    return _create_sample_prompts_file(dataset_name, dataset_path) 
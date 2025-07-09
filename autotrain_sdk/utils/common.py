"""Common utility functions shared between CLI and Gradio interfaces.

This module consolidates functions that were previously duplicated across different modules,
providing a single source of truth for common operations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gradio as gr
except ImportError:
    gr = None

from ..dataset import SUPPORTED_IMAGE_EXTS, INPUT_DIR, OUTPUT_DIR
from ..dataset_sources import find_dataset_path as _ds_find_path
from ..paths import get_project_root


__all__ = [
    "resolve_dataset_path",
    "list_available_datasets", 
    "dataset_file_counts",
    "load_integration_config",
    "save_integration_config",
    "initialize_integration_env_vars",
    "get_dataset_choices_for_ui",
]


###############################################################################
# Dataset Path Resolution
###############################################################################

def resolve_dataset_path(name: str) -> Optional[Path]:
    """Resolve dataset folder path, checking input directory first, then external sources.
    
    Args:
        name: Dataset name to resolve
        
    Returns:
        Path to dataset folder if found, None otherwise
    """
    # Check input directory first
    input_path = INPUT_DIR / name
    if input_path.exists():
        return input_path
    
    # Check external sources
    return _ds_find_path(name)


###############################################################################
# Dataset Listing
###############################################################################

def list_available_datasets() -> List[str]:
    """Get list of all available dataset names from input directory and external sources.
    
    Returns:
        Sorted list of dataset names
    """
    names = set()
    
    # Add datasets from input directory
    if INPUT_DIR.exists():
        names.update([f.name for f in INPUT_DIR.iterdir() if f.is_dir()])
    
    # Add external datasets
    from ..dataset_sources import list_external_datasets
    try:
        names.update(list_external_datasets().keys())
    except Exception:
        # If external datasets can't be loaded, continue with just input datasets
        pass
    
    return sorted(names)


def get_dataset_choices_for_ui(selected: Optional[str] = None) -> Any:
    """Get dataset choices formatted for Gradio UI dropdown.
    
    Args:
        selected: Currently selected dataset name
        
    Returns:
        Gradio update dict if gradio is available, otherwise just the list
    """
    names = list_available_datasets()
    
    if gr is not None:
        # Return Gradio update
        value = selected if selected in names else (names[0] if names else None)
        return gr.update(choices=names, value=value)
    else:
        # Return plain list if Gradio not available
        return names


###############################################################################
# Dataset File Counts
###############################################################################

def dataset_file_counts(name: str) -> Tuple[int, int, bool]:
    """Count image and text files in a dataset, and check if output exists.
    
    Args:
        name: Dataset name
        
    Returns:
        Tuple of (image_count, text_count, has_output)
    """
    folder = resolve_dataset_path(name)
    if not folder or not folder.exists():
        return 0, 0, False
    
    # Count image files
    num_imgs = 0
    for ext in SUPPORTED_IMAGE_EXTS:
        num_imgs += len(list(folder.glob(f"*.{ext}")))
    
    # Count text files
    num_txt = len(list(folder.glob("*.txt")))
    
    # Check if output directory exists
    has_output = (OUTPUT_DIR / name).exists()
    
    return num_imgs, num_txt, has_output


###############################################################################
# Integration Configuration
###############################################################################

def get_integrations_config_path() -> Path:
    """Get path to integrations configuration file."""
    return get_project_root() / ".gradio" / "integrations.json"


def load_integration_config() -> Dict[str, Any]:
    """Load integration configuration from file.
    
    Returns:
        Dictionary with integration configuration, empty dict if file doesn't exist
    """
    config_path = get_integrations_config_path()
    
    try:
        if config_path.exists():
            return json.loads(config_path.read_text())
    except Exception:
        pass
    
    return {}


def save_integration_config(config: Dict[str, Any]) -> None:
    """Save integration configuration to file and update environment variables.
    
    Args:
        config: Configuration dictionary to save
    """
    config_path = get_integrations_config_path()
    
    try:
        # Save to file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2))
    except Exception:
        pass
    
    # Update environment variables for current session
    for key, value in config.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


def initialize_integration_env_vars() -> None:
    """Initialize environment variables from config file if they're not already set.
    
    This should be called at program startup to ensure integration functions
    work correctly even before any config changes are made.
    """
    config = load_integration_config()
    
    # Only set environment variables that aren't already set
    for key, value in config.items():
        if key not in os.environ and value is not None:
            os.environ[key] = str(value)


###############################################################################
# Helper Functions for UI Components
###############################################################################

def format_dataset_status(name: str) -> str:
    """Format dataset status string for UI display.
    
    Args:
        name: Dataset name
        
    Returns:
        Formatted status string
    """
    imgs, txts, has_output = dataset_file_counts(name)
    output_status = "✅ output" if has_output else ""
    return f"{name}: {imgs} images · {txts} txt {output_status}".strip()


def get_dataset_summary(name: str) -> Dict[str, Any]:
    """Get comprehensive dataset summary information.
    
    Args:
        name: Dataset name
        
    Returns:
        Dictionary with dataset information
    """
    imgs, txts, has_output = dataset_file_counts(name)
    path = resolve_dataset_path(name)
    
    return {
        "name": name,
        "path": str(path) if path else None,
        "image_count": imgs,
        "text_count": txts,
        "has_output": has_output,
        "status": format_dataset_status(name),
    } 
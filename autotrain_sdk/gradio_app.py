from __future__ import annotations

"""Gradio Application (MVP – Phase 3).

Allows:
- Preparing datasets
- Editing TOML presets
- Launching training with log and image streaming
"""

import subprocess
import threading
import time
from pathlib import Path
from typing import List, Tuple, Generator, Optional
import shutil
import signal
import os
from datetime import datetime
import toml
import re
from itertools import chain
import json
import itertools
from PIL import Image

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from .dataset import (
    create_input_folders,
    populate_output_structure,
    populate_output_structure_single,
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_TEXT_EXTS,
    create_sample_prompts,
    create_sample_prompts_for_dataset,
)
from .configurator import (
    load_config,
    update_config,
    generate_presets,
    generate_presets_for_dataset,
)
from .paths import OUTPUT_DIR, INPUT_DIR, BATCH_CONFIG_DIR, get_project_root, compute_run_dir
from .job_manager import JOB_MANAGER, Job, JobStatus
from .dataset_sources import (
    load_sources as _ds_load_src,
    add_source as _ds_add_src,
    remove_source as _ds_remove_src,
    find_dataset_path as _ds_find_path,
)
from .experiments import create_experiment  # Phase 1

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

# Global process holder for cancel functionality
CURRENT_PROC: Optional[subprocess.Popen[str]] = None

STOP_TOKENS = {
    "a",
    "the",
    "and",
    "of",
    "in",
    "on",
    "with",
    "is",
    "to",
    "for",
    "at",
    "by",
    "from",
    "as",
    "an",
    "her",
    "his",
    "she",
    "he",
    "it",
    "its",
}

def _tail_process(cmd: List[str], img_dir: Path | None = None) -> Generator[Tuple[str, List[str]], None, None]:
    """Executes a command and yields logs and images.

    Yields tuples of (log, gallery_paths)
    """

    global CURRENT_PROC
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # new process group for easier termination
    )
    CURRENT_PROC = proc
    log_buffer: List[str] = []
    last_gallery: List[str] = []
    last_img_scan = 0.0

    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            log_buffer.append(line.rstrip("\n"))
        # Scan for images every 2 seconds
        now = time.time()
        if img_dir and now - last_img_scan > 2:
            last_img_scan = now
            imgs = sorted([str(p) for p in img_dir.glob("**/*.png")])[-12:]
            if imgs != last_gallery:
                last_gallery = imgs
        if line:
            yield ("\n".join(log_buffer[-400:]), last_gallery)
        if not line and proc.poll() is not None:
            break
    # Process finished – return final state
    returncode = proc.wait()
    CURRENT_PROC = None
    final_msg = f"\n[Process finished with code {returncode}]"
    log_buffer.append(final_msg)
    yield ("\n".join(log_buffer[-400:]), last_gallery)


# ---------------------------------------------------------------------------
# Dataset Callbacks
# ---------------------------------------------------------------------------

def cb_create_folders(names: str):
    names_list = [n.strip() for n in names.split(",") if n.strip()]
    if not names_list:
        return "Please enter at least one name.", gr.update()
    created = create_input_folders(names_list)
    last_created = created[-1].name if created else None
    return (
        f"Folders created: {', '.join(p.name for p in created)}",
        _list_dataset_choices(last_created),
    )


def cb_build_output(min_imgs: int):
    try:
        populate_output_structure(min_images=int(min_imgs))
        return "Output structure 'output/' created and files copied."
    except Exception as e:
        return f"Error: {e}"


def cb_create_sample_prompts():
    """Callback to create sample_prompts.txt files for all datasets."""
    try:
        count = create_sample_prompts()
        if count > 0:
            return f"✓ Created/updated {count} sample_prompts.txt files"
        else:
            return "⚠️ No datasets found in output/ directory. Please create output structure first."
    except Exception as e:
        return f"❌ Error: {e}"


def cb_create_sample_prompts_single(dataset_name: str):
    """Callback to create a sample_prompts.txt file for a specific dataset."""
    if not dataset_name or not dataset_name.strip():
        return "❌ Please select a dataset first"
    
    try:
        success = create_sample_prompts_for_dataset(dataset_name.strip())
        if success:
            return f"✓ Created sample_prompts.txt for dataset '{dataset_name}'"
        else:
            return f"❌ Failed to create sample_prompts.txt for dataset '{dataset_name}'. Make sure the dataset exists in output/"
    except Exception as e:
        return f"❌ Error: {e}"


# ---------------------------------------------------------------------------
# Configuration Callbacks
# ---------------------------------------------------------------------------

def cb_load_config(file_obj):
    if file_obj is None:
        return None, "Select a TOML file"
    cfg_path = Path(file_obj.name) if hasattr(file_obj, "name") else Path(file_obj)
    cfg = load_config(cfg_path)
    rows = [[k, str(v)] for k, v in cfg.items()]
    return rows, f"Loaded {cfg_path}"


def cb_save_config(file_obj, table):
    if file_obj is None:
        return "Select a TOML file"  # type: ignore[return-value]
    cfg_path = Path(file_obj.name) if hasattr(file_obj, "name") else Path(file_obj)
    updates = {row[0]: row[1] for row in table if row and row[0]}
    update_config(cfg_path, updates)
    return f"Saved {cfg_path}"  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------------

def cb_start_training(profile: str, dataset_name):
    """Inicia el entrenamiento y muestra la ruta final del modelo."""

    if not dataset_name:
        return "Select a dataset", "Select a dataset", []

    # auto-import si el dataset vive fuera de input/
    import shutil
    if not (INPUT_DIR / dataset_name).exists():
        ext_path = _ds_find_path(dataset_name)
        if ext_path:
            dest = INPUT_DIR / dataset_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ext_path, dest, dirs_exist_ok=True)

    cfg_path = BATCH_CONFIG_DIR / (
        "Flux" if profile == "Flux" else ("FluxLORA" if profile == "FluxLORA" else "Nude")
    ) / f"{dataset_name}.toml"

    # Determinar run_dir usando la misma lógica que en el CLI (paths.compute_run_dir)
    run_dir = compute_run_dir(dataset_name, profile)

    # Si la carpeta no existe aún (p.ej. AUTO_REMOTE_DIRECT==1), créala para poder escribir config
    run_dir.mkdir(parents=True, exist_ok=True)

    # Generar config parcheada
    orig_cfg = toml.load(cfg_path)
    orig_cfg["output_dir"] = str(run_dir)
    orig_cfg["logging_dir"] = str(run_dir / "log")

    patched_cfg_path = run_dir / "config.toml"
    with patched_cfg_path.open("w", encoding="utf-8") as f:
        toml.dump(orig_cfg, f)

    # Encolar job
    job = Job(dataset_name, profile, patched_cfg_path, run_dir)
    JOB_MANAGER.enqueue(job)

    path_msg = f"**Output dir → `{run_dir}`**"
    log_msg = f"Job {job.id} queued ({dataset_name}-{profile})"

    return path_msg, log_msg, []


def cb_start_batch_training(profile: str, selected_datasets: list):
    """Inicia entrenamiento en batch para múltiples datasets."""
    
    if not selected_datasets:
        return "No datasets selected", "No datasets selected", []
    
    import shutil
    
    queued_jobs = []
    
    for dataset_name in selected_datasets:
        try:
            # auto-import si el dataset vive fuera de input/
            if not (INPUT_DIR / dataset_name).exists():
                ext_path = _ds_find_path(dataset_name)
                if ext_path:
                    dest = INPUT_DIR / dataset_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(ext_path, dest, dirs_exist_ok=True)

            cfg_path = BATCH_CONFIG_DIR / (
                "Flux" if profile == "Flux" else ("FluxLORA" if profile == "FluxLORA" else "Nude")
            ) / f"{dataset_name}.toml"

            # Determinar run_dir usando la misma lógica que en el CLI (paths.compute_run_dir)
            run_dir = compute_run_dir(dataset_name, profile)

            # Si la carpeta no existe aún (p.ej. AUTO_REMOTE_DIRECT==1), créala para poder escribir config
            run_dir.mkdir(parents=True, exist_ok=True)

            # Generar config parcheada
            orig_cfg = toml.load(cfg_path)
            orig_cfg["output_dir"] = str(run_dir)
            orig_cfg["logging_dir"] = str(run_dir / "log")

            patched_cfg_path = run_dir / "config.toml"
            with patched_cfg_path.open("w", encoding="utf-8") as f:
                toml.dump(orig_cfg, f)

            # Encolar job
            job = Job(dataset_name, profile, patched_cfg_path, run_dir)
            JOB_MANAGER.enqueue(job)
            queued_jobs.append(job.id)
            
        except Exception as e:
            queued_jobs.append(f"Error with {dataset_name}: {str(e)}")
    
    path_msg = f"**Batch training queued for {len(selected_datasets)} datasets**"
    log_msg = f"Batch jobs queued: {', '.join(str(job) for job in queued_jobs)}"

    return path_msg, log_msg, []


# ---------------------------------------------------------------------------
# Building the UI
# ---------------------------------------------------------------------------

CSS_WRAP = """
+.gr-dataframe tbody td,
+.ag-cell {
  white-space: normal !important;
  word-break: break-word !important;
  line-height: 1.3;
}
"""

def _parse_res(res_str: str) -> tuple[int, int] | None:
    """Parse a resolution string like '512x768' -> (512,768). Return None if invalid or '0x0'."""
    if not res_str:
        return None
    try:
        if "x" in res_str:
            w_str, h_str = res_str.lower().split("x", 1)
            w, h = int(w_str), int(h_str)
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    return None


def _crop_dataset(dataset: str, res_str: str) -> int:
    """Resize/crop images in input/<dataset> to *res_str* (WxH). Returns number of files processed."""
    target = _parse_res(res_str)
    if target is None:
        return 0
    w_t, h_t = target
    folder = INPUT_DIR / dataset
    if not folder.exists():
        return 0
    processed = 0
    for p in folder.iterdir():
        if p.suffix.lower().lstrip(".") not in SUPPORTED_IMAGE_EXTS:
            continue
        try:
            with Image.open(p) as img:
                # center crop preserving aspect then resize
                img_w, img_h = img.size
                # Calculate crop rectangle
                aspect_target = w_t / h_t
                aspect_img = img_w / img_h
                if aspect_img > aspect_target:
                    # wider -> crop width
                    new_w = int(aspect_target * img_h)
                    left = (img_w - new_w) // 2
                    box = (left, 0, left + new_w, img_h)
                else:
                    # taller -> crop height
                    new_h = int(img_w / aspect_target)
                    top = (img_h - new_h) // 2
                    box = (0, top, img_w, top + new_h)
                cropped = img.crop(box).resize((w_t, h_t), Image.LANCZOS)
                cropped.save(p)
                processed += 1
        except Exception:
            pass
    return processed


def _crop_images_in_folder(folder: Path, res_str: str) -> int:
    """Crop/resize every supported image in *folder* to *res_str* (WxH). Returns number processed."""
    target = _parse_res(res_str)
    if target is None or not folder.exists():
        return 0
    w_t, h_t = target
    processed = 0
    for p in folder.iterdir():
        if p.suffix.lower().lstrip('.') not in SUPPORTED_IMAGE_EXTS:
            continue
        try:
            with Image.open(p) as img:
                img_w, img_h = img.size
                aspect_target = w_t / h_t
                aspect_img = img_w / img_h
                if aspect_img > aspect_target:
                    new_w = int(aspect_target * img_h)
                    left = (img_w - new_w) // 2
                    box = (left, 0, left + new_w, img_h)
                else:
                    new_h = int(img_w / aspect_target)
                    top = (img_h - new_h) // 2
                    box = (0, top, img_w, top + new_h)
                img.crop(box).resize((w_t, h_t), Image.LANCZOS).save(p)
                processed += 1
        except Exception:
            pass
    return processed


def _get_dataset_info(dataset_name: str) -> dict:
    """Get detailed information about a dataset."""
    if not dataset_name:
        return {
            "exists": False,
            "error": "No dataset selected"
        }
    
    # Check in output directory first (prioritized)
    input_path = INPUT_DIR / dataset_name
    output_path = OUTPUT_DIR / dataset_name
    
    dataset_path = None
    if output_path.exists():
        dataset_path = output_path / "img" / f"30_{dataset_name} person"
        if not dataset_path.exists():
            # Try to find the actual folder
            for subdir in output_path.iterdir():
                if subdir.is_dir() and subdir.name.startswith("img"):
                    for imgdir in subdir.iterdir():
                        if imgdir.is_dir() and dataset_name in imgdir.name:
                            dataset_path = imgdir
                            break
                    break
    elif input_path.exists():
        dataset_path = input_path
    else:
        # Try external sources
        ext_path = _ds_find_path(dataset_name)
        if ext_path:
            dataset_path = ext_path
    
    if not dataset_path or not dataset_path.exists():
        return {
            "exists": False,
            "error": f"Dataset '{dataset_name}' not found"
        }
    
    try:
        # Count images and get info
        image_files = []
        text_files = []
        total_size = 0
        
        for file_path in dataset_path.iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size
                
                if file_path.suffix.lower().lstrip('.') in SUPPORTED_IMAGE_EXTS:
                    image_files.append(file_path)
                elif file_path.suffix.lower().lstrip('.') in SUPPORTED_TEXT_EXTS:
                    text_files.append(file_path)
        
        # Get image resolution info
        resolutions = []
        if image_files:
            # Sample up to 5 images to get resolution info
            sample_images = image_files[:5]
            for img_path in sample_images:
                try:
                    with Image.open(img_path) as img:
                        resolutions.append((img.width, img.height))
                except Exception:
                    continue
        
        # Calculate average resolution
        avg_resolution = "N/A"
        if resolutions:
            avg_w = sum(r[0] for r in resolutions) / len(resolutions)
            avg_h = sum(r[1] for r in resolutions) / len(resolutions)
            avg_resolution = f"{int(avg_w)}x{int(avg_h)}"
        
        # Format total size
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{total_size / (1024 * 1024 * 1024):.1f} GB"
        
        # Get last modified date
        last_modified = max(f.stat().st_mtime for f in dataset_path.iterdir() if f.is_file())
        last_modified_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M")
        
        return {
            "exists": True,
            "num_images": len(image_files),
            "num_captions": len(text_files),
            "total_size": size_str,
            "avg_resolution": avg_resolution,
            "last_modified": last_modified_str,
            "path": str(dataset_path),
            "sample_images": [str(f) for f in image_files[:4]]  # First 4 images for preview
        }
        
    except Exception as e:
        return {
            "exists": False,
            "error": f"Error analyzing dataset: {str(e)}"
        }


def _estimate_training_time(dataset_name: str, profile: str) -> dict:
    """Estimate training time based on dataset size, profile, and active GPU."""
    if not dataset_name:
        return {
            "error": "No dataset selected"
        }
    
    # Get dataset info
    dataset_info = _get_dataset_info(dataset_name)
    if not dataset_info.get("exists", False):
        return {
            "error": "Dataset not found"
        }
    
    num_images = dataset_info.get("num_images", 0)
    if num_images == 0:
        return {
            "error": "No images found in dataset"
        }
    
    # Get active GPU profile for realistic timing
    gpu_profile = _get_active_gpu_profile()
    
    # Get time per iteration from GPU profile (in seconds)
    profile_speeds = {
        "Flux": gpu_profile.get("flux_speed", 0.8),
        "FluxLORA": gpu_profile.get("flux_lora_speed", 0.5),
        "Nude": gpu_profile.get("nude_speed", 0.6)
    }
    
    seconds_per_iteration = profile_speeds.get(profile, 0.6)
    
    # Load config to get epochs and other parameters
    try:
        cfg_path = BATCH_CONFIG_DIR / (
            "Flux" if profile == "Flux" else 
            ("FluxLORA" if profile == "FluxLORA" else "Nude")
        ) / f"{dataset_name}.toml"
        
        if cfg_path.exists():
            config = toml.load(cfg_path)
            max_epochs = config.get("max_train_epochs", 10)
            batch_size = config.get("train_batch_size", 1)
            repeats = config.get("dataset_repeats", 30)
        else:
            # Default values
            max_epochs = 10
            batch_size = 1
            repeats = 30
    except Exception:
        max_epochs = 10
        batch_size = 1
        repeats = 30
    
    # Calculate total iterations
    total_steps = (num_images * repeats * max_epochs) // batch_size
    
    # Estimate time in seconds, then convert to minutes
    estimated_seconds = total_steps * seconds_per_iteration
    estimated_minutes = estimated_seconds / 60
    
    # Format time
    if estimated_minutes < 60:
        time_str = f"{int(estimated_minutes)} min"
    elif estimated_minutes < 1440:  # 24 hours
        hours = int(estimated_minutes // 60)
        mins = int(estimated_minutes % 60)
        time_str = f"{hours}h {mins}m"
    else:
        days = int(estimated_minutes // 1440)
        hours = int((estimated_minutes % 1440) // 60)
        time_str = f"{days}d {hours}h"
    
    # Calculate power consumption estimate
    power_consumption = gpu_profile.get("power_consumption", 300)  # Watts
    estimated_power_kwh = (estimated_seconds / 3600) * (power_consumption / 1000)
    
    return {
        "estimated_time": time_str,
        "estimated_minutes": int(estimated_minutes),
        "estimated_seconds": int(estimated_seconds),
        "total_steps": total_steps,
        "gpu_name": gpu_profile.get("name", "Unknown GPU"),
        "power_consumption_kwh": round(estimated_power_kwh, 2),
        "parameters": {
            "epochs": max_epochs,
            "batch_size": batch_size,
            "repeats": repeats
        }
    }


def _create_metrics_plots(metrics_data: dict) -> tuple[str, str, str, str]:
    """Create enhanced metrics plots for training visualization."""
    if not metrics_data:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            x=0.5, y=0.5, text="No data available",
            xref="paper", yref="paper",
            showarrow=False, font_size=16
        )
        empty_fig.update_layout(
            showlegend=False,
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=280,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        # Generate wrapped HTML with fixed heights
        base_empty_html = empty_fig.to_html(include_plotlyjs="cdn")
        loss_empty = f'<div style="height: 300px; overflow: hidden;">{empty_fig.to_html(include_plotlyjs="cdn", div_id="loss_plot")}</div>'
        progress_empty = f'<div style="height: 200px; overflow: hidden;">{empty_fig.to_html(include_plotlyjs="cdn", div_id="progress_plot")}</div>'
        
        return loss_empty, base_empty_html, base_empty_html, progress_empty
    
    # Extract data
    history = metrics_data.get("history_data", {})
    loss_history = history.get("loss_history", [])
    lr_history = history.get("lr_history", [])
    memory_history = history.get("memory_history", [])
    
    # 1. Loss Curves Plot
    loss_fig = go.Figure()
    
    if loss_history:
        steps = [point["step"] for point in loss_history]
        
        # Train loss
        train_losses = [point.get("train_loss") for point in loss_history if point.get("train_loss") is not None]
        train_steps = [point["step"] for point in loss_history if point.get("train_loss") is not None]
        if train_losses:
            loss_fig.add_trace(go.Scatter(
                x=train_steps, y=train_losses,
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='#ff6b6b', width=2),
                marker=dict(size=4)
            ))
        
        # Validation loss
        val_losses = [point.get("val_loss") for point in loss_history if point.get("val_loss") is not None]
        val_steps = [point["step"] for point in loss_history if point.get("val_loss") is not None]
        if val_losses:
            loss_fig.add_trace(go.Scatter(
                x=val_steps, y=val_losses,
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='#4ecdc4', width=2),
                marker=dict(size=4)
            ))
        
        # Average loss
        avg_losses = [point.get("avg_loss") for point in loss_history if point.get("avg_loss") is not None]
        avg_steps = [point["step"] for point in loss_history if point.get("avg_loss") is not None]
        if avg_losses:
            loss_fig.add_trace(go.Scatter(
                x=avg_steps, y=avg_losses,
                mode='lines',
                name='Average Loss',
                line=dict(color='#45b7d1', width=2, dash='dash')
            ))
    
    loss_fig.update_layout(
        title="Training Loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        height=280,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(x=0.7, y=0.95),
        template="plotly_white"
    )
    
    # 2. Learning Rate Schedule
    lr_fig = go.Figure()
    
    if lr_history:
        lr_steps = [point["step"] for point in lr_history]
        learning_rates = [point["lr"] for point in lr_history]
        
        lr_fig.add_trace(go.Scatter(
            x=lr_steps, y=learning_rates,
            mode='lines+markers',
            name='Learning Rate',
            line=dict(color='#ffa500', width=2),
            marker=dict(size=4)
        ))
    
    lr_fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        yaxis_type="log",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white"
    )
    
    # 3. System Metrics (Memory & GPU)
    system_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Memory Usage (MB)', 'GPU Utilization (%)'),
        vertical_spacing=0.15
    )
    
    if memory_history:
        mem_steps = [point["step"] for point in memory_history]
        memory_used = [point.get("memory_used", 0) for point in memory_history]
        gpu_util = [point.get("gpu_util", 0) for point in memory_history]
        
        # Memory usage
        system_fig.add_trace(go.Scatter(
            x=mem_steps, y=memory_used,
            mode='lines+markers',
            name='Memory Used',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=3)
        ), row=1, col=1)
        
        # GPU utilization
        system_fig.add_trace(go.Scatter(
            x=mem_steps, y=gpu_util,
            mode='lines+markers',
            name='GPU Utilization',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=3)
        ), row=2, col=1)
    
    system_fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        showlegend=False
    )
    
    # 4. Progress Overview
    basic_metrics = metrics_data.get("basic_metrics", {})
    training_metrics = metrics_data.get("training_metrics", {})
    system_metrics = metrics_data.get("system_metrics", {})
    
    progress_fig = go.Figure()
    
    # Progress bar
    progress = basic_metrics.get("progress", 0)
    progress_fig.add_trace(go.Bar(
        x=[progress, 100-progress],
        y=['Progress'],
        orientation='h',
        marker=dict(color=['#2ecc71', '#ecf0f1']),
        showlegend=False,
        text=[f'{progress}%', ''],
        textposition='inside'
    ))
    
    progress_fig.update_layout(
        title=f"Training Progress - Step {basic_metrics.get('current_step', 0):,} / {basic_metrics.get('total_steps', 0):,}",
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=180,
        margin=dict(l=40, r=40, t=40, b=10),
        template="plotly_white"
    )
    
    # Convert to HTML with fixed height containers
    loss_html = f'<div style="height: 300px; overflow: hidden;">{loss_fig.to_html(include_plotlyjs="cdn", div_id="loss_plot")}</div>'
    lr_html = lr_fig.to_html(include_plotlyjs='cdn')  # Not used in new layout
    system_html = system_fig.to_html(include_plotlyjs='cdn')  # Not used in new layout
    progress_html = f'<div style="height: 200px; overflow: hidden;">{progress_fig.to_html(include_plotlyjs="cdn", div_id="progress_plot")}</div>'
    
    return loss_html, lr_html, system_html, progress_html


def _format_enhanced_metrics(metrics_data: dict) -> str:
    """Format enhanced metrics data for display."""
    if not metrics_data:
        return "⚠️ No metrics data available"
    
    basic = metrics_data.get("basic_metrics", {})
    training = metrics_data.get("training_metrics", {})
    system = metrics_data.get("system_metrics", {})
    timestamps = metrics_data.get("timestamps", {})
    
    # Check if we have any progress information
    has_progress = (basic.get("current_step", 0) > 0 or 
                   basic.get("progress", 0) > 0 or 
                   basic.get("total_steps", 0) > 0)
    
    # Check if we have any loss information
    has_loss = (training.get("avg_loss") is not None or 
               training.get("train_loss") is not None or 
               training.get("val_loss") is not None)
    
    if not has_progress and not has_loss:
        return "⏳ Training starting... (no metrics yet)"
    
    # Calculate runtime
    runtime_str = "N/A"
    if timestamps.get("start_time") and timestamps.get("last_update"):
        runtime_seconds = timestamps["last_update"] - timestamps["start_time"]
        if runtime_seconds > 3600:
            hours = int(runtime_seconds // 3600)
            minutes = int((runtime_seconds % 3600) // 60)
            runtime_str = f"{hours}h {minutes}m"
        elif runtime_seconds > 60:
            minutes = int(runtime_seconds // 60)
            seconds = int(runtime_seconds % 60)
            runtime_str = f"{minutes}m {seconds}s"
        else:
            runtime_str = f"{int(runtime_seconds)}s"
    
    lines = []
    
    # Always show progress if we have it
    if has_progress:
        lines.extend([
            "**📈 Progress**",
            f"• **Step:** {basic.get('current_step', 0):,} / {basic.get('total_steps', 0):,}",
            f"• **Progress:** {basic.get('progress', 0):.1f}%",
            f"• **ETA:** {basic.get('eta', 'N/A')}",
            f"• **Runtime:** {runtime_str}",
            f"• **Speed:** {basic.get('rate', 0):.2f} s/it",
            ""
        ])
    
    # Show training metrics if we have loss information
    if has_loss or training.get('learning_rate', 0) > 0:
        lines.append("**🎯 Training Metrics**")
        
        # Learning rate
        lr = training.get('learning_rate', 0)
        if lr > 0:
            lines.append(f"• **Learning Rate:** {lr:.2e}")
        
        # Losses - prioritize avg_loss since it's most commonly available
        avg_loss = training.get('avg_loss')
        train_loss = training.get('train_loss')
        val_loss = training.get('val_loss')
        
        if avg_loss is not None:
            lines.append(f"• **Avg Loss:** {avg_loss:.4f}")
        if train_loss is not None:
            lines.append(f"• **Train Loss:** {train_loss:.4f}")
        if val_loss is not None:
            lines.append(f"• **Val Loss:** {val_loss:.4f}")
        
        # Gradient norm
        grad_norm = training.get('grad_norm')
        if grad_norm is not None:
            lines.append(f"• **Grad Norm:** {grad_norm:.4f}")
        
        lines.append("")
    
    # Add system metrics if available
    memory_used = system.get('memory_used', 0)
    gpu_util = system.get('gpu_utilization', 0)
    temp = system.get('temperature', 0)
    
    if memory_used > 0 or gpu_util > 0 or temp > 0:
        lines.extend([
            "**🔧 System Metrics**",
            f"• **Memory Used:** {memory_used:.1f} MB" if memory_used > 0 else "• **Memory Used:** N/A",
            f"• **GPU Util:** {gpu_util:.1f}%" if gpu_util > 0 else "• **GPU Util:** N/A",
            f"• **Temperature:** {temp:.1f}°C" if temp > 0 else "• **Temperature:** N/A",
        ])
    
    # If we have no meaningful data to show
    if not lines:
        return "⏳ Training starting... (waiting for metrics)"
    
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AutoTrainV2", css=CSS_WRAP) as demo:
        gr.Markdown("# AutoTrainV2 – Dataset preparation / Model Training: Flux - SDXL")

        # ------------------------------------------------------------------
        # Shared helper functions for external dataset sources (used in both
        # Create and Manage sub-tabs). Moved here so they are defined before
        # the UI elements that reference them.
        # ------------------------------------------------------------------

        def _fmt_src(rows):
            return {"headers": ["Path"], "data": rows}

        def _refresh_src_table():
            return _fmt_src([[str(p)] for p in _ds_load_src()])

        def _add_src(path):
            if not path:
                return _refresh_src_table(), "Enter path", _refresh_src_dropdown()
            _ds_add_src(path)
            return _refresh_src_table(), f"Added {path}", _refresh_src_dropdown()

        def _extract_rows_any(data):
            if data is None:
                return []
            if isinstance(data, dict):
                return data.get("data", [])
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    return data.values.tolist()
            except Exception:
                pass
            return data if isinstance(data, list) else []

        def _del_src(df_payload):
            rows = _extract_rows_any(df_payload)
            if not rows:
                return _refresh_src_table(), "Select a row", _refresh_src_dropdown()
            for row in rows:
                if row and row[0]:
                    _ds_remove_src(row[0])
            return _refresh_src_table(), "Removed", _refresh_src_dropdown()

        def _refresh_src_dropdown():
            return gr.update(choices=[str(p) for p in _ds_load_src()])

        def _process_source(src_path_str: str):
            from pathlib import Path
            if not src_path_str:
                return _refresh_src_table(), "Select source first"

            src_path = Path(src_path_str)
            if not src_path.exists():
                return _refresh_src_table(), f"Path not found: {src_path}"

            datasets = [d.name for d in src_path.iterdir() if d.is_dir()]
            if not datasets:
                return _refresh_src_table(), "No datasets in source"

            success_cnt = 0
            for name in datasets:
                try:
                    cb_build_output_single(name)
                    generate_presets_for_dataset(name)
                    success_cnt += 1
                except Exception:
                    pass

            return _refresh_src_table(), f"Processed {success_cnt} / {len(datasets)} datasets"

        with gr.Tab("Dataset"):

            with gr.Tabs():
                # -------- CREATE Tab --------
                with gr.TabItem("Create"):
                    gr.Markdown("### Create new dataset")
                    names = gr.Textbox(label="Name(s) (comma-separated)")
                    create_btn = gr.Button("Create")
                    create_out = gr.Markdown()
                    create_btn.click(lambda txt: cb_create_folders(txt)[0], inputs=names, outputs=create_out)

                    # ---- Dataset sources (external folders) ----
                    with gr.Accordion("Dataset sources (external folders)", open=False):
                        gr.Markdown("""### External dataset folders

Use this section when your datasets live on another drive or shared storage.

1. **Add source** – paste the absolute path that contains many dataset sub-folders.
2. (Optional) **Remove selected** – delete a path from the registry.
3. **Process source** – choose a path and click the orange button:
   – Copies every dataset to `input/` (if not present).
   – Creates/updates the required `output/<dataset>/...` structure.
   – Generates default TOML presets.

After processing, the new datasets are available in *Manage*, *Training* and *Experiments* tabs.
""")

                        src_list_c = gr.Dataframe(headers=["Path"], interactive=False, row_count=(5, "dynamic"))
                        new_src_path_c = gr.Textbox(label="Add source path (folder)")
                        add_src_btn_c = gr.Button("Add source")
                        del_src_btn_c = gr.Button("Remove selected")
                        src_select_c = gr.Dropdown(label="Select source to process", choices=[str(p) for p in _ds_load_src()])
                        proc_src_btn_c = gr.Button("Import + Build + Presets for all datasets in source", variant="primary")

                        # Callbacks (reuse helper fns from Manage)
                        add_src_btn_c.click(_add_src, inputs=new_src_path_c, outputs=[src_list_c, create_out, src_select_c])
                        del_src_btn_c.click(_del_src, inputs=src_list_c, outputs=[src_list_c, create_out, src_select_c])
                        proc_src_btn_c.click(_process_source, inputs=src_select_c, outputs=[src_list_c, create_out])

                        # init values
                        src_list_c.value = {"headers":["Path"],"data":[[str(p)] for p in _ds_load_src()]}
                        src_select_c.choices = [str(p) for p in _ds_load_src()]

                # -------- Pestaña GESTIONAR --------
                with gr.TabItem("Manage"):
                    gr.Markdown("### Manage existing dataset")

                    # --- Layout: controls (left) | preview / captions (right) ---
                    with gr.Row():
                        # LEFT column – adjustments & tools
                        with gr.Column(scale=4, min_width=260):
                            # ---- Dataset selector row ----
                            with gr.Row():
                                ds_dropdown = gr.Dropdown(label="Select dataset", choices=_list_dataset_raw(), scale=6)
                                refresh_ds_btn = gr.Button("↻", scale=1)

                            ds_status = gr.Markdown()

                            uploader = gr.File(
                                label="Upload images (.png/.jpg) and .txt",
                                file_types=["image", ".txt"],
                                file_count="multiple",
                                visible=False,
                            )
                            upload_out = gr.Markdown()

                            # Dataset statistics / info
                            ds_stats_md = gr.Markdown()

                            # Slider to choose number of repetitions when creating the output structure
                            reps_slider = gr.Slider(
                                label="Repetitions",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=30,
                                scale=2,
                            )

                            res_txt = gr.Textbox(
                                label="Crop resolution (WxH)",
                                placeholder="e.g. 512x512 (blank = no crop)",
                                value="",
                                scale=2,
                            )

                            # (Los botones se moverán al final para mejor claridad)
                            del_out = gr.Markdown()
                            build_out = gr.Markdown()
                            scan_out = gr.Markdown()

                            # ---- Captioning (rename + generate captions) ----
                            with gr.Accordion("Captioning", open=False):
                                gr.Markdown("Rename images using a trigger word and create caption .txt files with Florence-2")
                                trig_txt = gr.Textbox(label="Trigger / base name", placeholder="e.g. laura")
                                cap_overwrite = gr.Checkbox(label="Overwrite existing captions", value=True)
                                cap_max = gr.Slider(label="Max tokens", minimum=32, maximum=512, step=32, value=128)
                                prep_btn = gr.Button("Rename + Generate captions", variant="primary")
                                prep_out = gr.Markdown()

                            # Dataset Sources accordion
                            with gr.Accordion("Dataset sources (external folders)", open=False, visible=False):
                                gr.Markdown("""### External dataset folders

Use this section when your datasets live on another drive or shared storage.

1. **Add source** – paste the absolute path that contains many dataset sub-folders.
2. (Optional) **Remove selected** – delete a path from the registry.
3. **Process source** – choose a path and click the orange button:
   – Copies every dataset to `input/` (if not present).
   – Creates/updates the required `output/<dataset>/...` structure.
   – Generates default TOML presets.

After processing, the new datasets are available in *Manage*, *Training* and *Experiments* tabs.
""")
                                src_list = gr.Dataframe(headers=["Path"], interactive=False, row_count=(5, "dynamic"))
                                new_src_path = gr.Textbox(label="Add source path (folder)")
                                add_src_btn = gr.Button("Add source")
                                del_src_btn = gr.Button("Remove selected")
                                src_select = gr.Dropdown(label="Select source to process", choices=[str(p) for p in _ds_load_src()])
                                proc_src_btn = gr.Button("Import + Build + Presets for all datasets in source", variant="primary")

                        # RIGHT column – preview & caption table
                        with gr.Column(scale=8):
                            with gr.Tabs():
                                with gr.TabItem("Preview"):
                                    preview_gallery = gr.Gallery(label="Preview", columns=4, visible=False)
                                with gr.TabItem("Captions"):
                                    file_table = gr.Dataframe(headers=["File", "Caption"], interactive=False, visible=False)
                            # Note: callbacks wiring remains unchanged below.

                    # End Row (layout)

                    # ---------------- Callback wiring (Manage tab) ----------------

                    # Refresh dataset list in dropdown
                    refresh_ds_btn.click(lambda: _list_dataset_choices(), inputs=[], outputs=ds_dropdown)

                    # -- Helper to rebuild preview, captions table and stats --
                    def _cb_select_ds(name):
                        up_vis, status = cb_dataset_selected(name)
                        if not name:
                            return up_vis, status, gr.update(visible=False), "", gr.update(visible=False)

                        imgs = []          # gallery content
                        rows = []          # table content
                        captions_tokens = []

                        folder = _resolve_dataset_path(name)
                        if folder is None:
                            return up_vis, status, gr.update(visible=False), "", gr.update(visible=False)

                        paths = []
                        for ext in SUPPORTED_IMAGE_EXTS:
                            paths.extend(folder.glob(f"*.{ext}"))
                        paths = sorted(paths)[:24]   # limit preview

                        from PIL import Image
                        from collections import Counter

                        for p in paths:
                            caption = ""
                            txt = p.with_suffix(".txt")
                            if txt.exists():
                                caption = txt.read_text("utf-8").strip()
                            imgs.append([str(p), caption])
                            rows.append([p.name, caption])
                            captions_tokens.extend([tok.lower() for tok in caption.split() if len(tok) >= 4 and tok.lower() not in STOP_TOKENS])

                        # stats
                        n_imgs = len(list(folder.glob("*.png"))) + len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.jpeg")))
                        if n_imgs:
                            sizes = []
                            for p in paths:
                                try:
                                    with Image.open(p) as im:
                                        sizes.append(im.size)
                                except Exception:
                                    pass
                            if sizes:
                                avg_w = int(sum(s[0] for s in sizes) / len(sizes))
                                avg_h = int(sum(s[1] for s in sizes) / len(sizes))
                            else:
                                avg_w = avg_h = 0
                        else:
                            avg_w = avg_h = 0

                        common_tokens = Counter(captions_tokens).most_common(10)
                        tokens_str = ", ".join(f"{tok}({cnt})" for tok, cnt in common_tokens)
                        stats_txt = f"**Images:** {n_imgs} · **Avg reso:** {avg_w}x{avg_h} · **Top tokens:** {tokens_str}"

                        return (
                            up_vis,                           # uploader visibility
                            status,                            # status markdown
                            gr.update(visible=True, value=rows),       # captions table
                            gr.update(visible=True, value=imgs),       # gallery
                            gr.update(visible=True, value=stats_txt),  # stats markdown
                        )

                    # Dropdown change → update preview, table, stats
                    ds_dropdown.change(
                        _cb_select_ds,
                        inputs=ds_dropdown,
                        outputs=[uploader, ds_status, file_table, preview_gallery, ds_stats_md],
                    )

                    # ----- Upload files -----
                    def _upload_and_refresh(dataset, files):
                        out_msg, status_msg = cb_upload_files(dataset, files)
                        # refresh preview after upload
                        _vis, _stat, table_upd, gal_upd, stats_upd = _cb_select_ds(dataset)
                        return out_msg, status_msg, table_upd, gal_upd, stats_upd

                    uploader.upload(
                        _upload_and_refresh,
                        inputs=[ds_dropdown, uploader],
                        outputs=[upload_out, ds_status, file_table, preview_gallery, ds_stats_md],
                    )

                    # ----- Duplicate / corrupt scan -----
                    def _scan_ds(dataset):
                        if not dataset:
                            return "Select dataset first"
                        folder = _resolve_dataset_path(dataset)
                        if folder is None:
                            return "Dataset not found"

                        from PIL import Image, UnidentifiedImageError
                        import hashlib

                        hashes = {}
                        dup_count = 0
                        corrupt_count = 0
                        img_paths = chain(folder.glob("*.png"), folder.glob("*.jpg"), folder.glob("*.jpeg"))
                        for p in img_paths:
                            try:
                                with Image.open(p) as im:
                                    im.verify()
                                h = hashlib.md5(p.read_bytes()).hexdigest()
                                if h in hashes:
                                    dup_count += 1
                                else:
                                    hashes[h] = p
                            except (UnidentifiedImageError, OSError):
                                corrupt_count += 1
                        return f"Duplicates: {dup_count} · Corrupt: {corrupt_count}"

                    # (wiring moved below action buttons)
                    # scan_btn.click(_scan_ds, inputs=ds_dropdown, outputs=scan_out)

                    # ----- Captioning callback -----

                    def _prep_ds(ds, trigger, overwrite, max_tok):
                        if not ds or not trigger.strip():
                            return "Select dataset and trigger first", gr.update(), gr.update(), gr.update()
                        from .captioner import rename_and_caption_dataset
                        try:
                            n = rename_and_caption_dataset(ds, trigger.strip(), overwrite=overwrite, max_new_tokens=int(max_tok))
                            msg = f"{n} caption(s) generated and images renamed."
                        except Exception as e:
                            msg = f"Error: {e}"
                        _vis, _st, tbl, gal, stats = _cb_select_ds(ds)
                        return msg, tbl, gal, stats

                    prep_btn.click(
                        _prep_ds,
                        inputs=[ds_dropdown, trig_txt, cap_overwrite, cap_max],
                        outputs=[prep_out, file_table, preview_gallery, ds_stats_md],
                    )

                    # ----- Delete images helper -----
                    def _delete_and_refresh(dataset):
                        if not dataset:
                            return "Select dataset first", gr.update(), gr.update(), gr.update(), gr.update()
                        folder = _resolve_dataset_path(dataset)
                        if folder is None or not folder.exists():
                            return "Dataset not found", gr.update(), gr.update(), gr.update(), gr.update()

                        count = 0
                        for ext in SUPPORTED_IMAGE_EXTS:
                            for p in folder.glob(f"*.{ext}"):
                                try:
                                    p.unlink()
                                    count += 1
                                except Exception:
                                    pass

                        msg = f"Deleted {count} image(s) from '{dataset}'."

                        # Refresh dataset preview & stats
                        _vis, status, table_upd, gal_upd, stats_upd = _cb_select_ds(dataset)
                        return msg, status, table_upd, gal_upd, stats_upd

                    # (wiring moved below action buttons)
                    # del_btn.click(lambda: None, [], [], js="() => confirm('Delete ALL images in this dataset folder?')").then(
                    #     _delete_and_refresh,
                    #     inputs=ds_dropdown,
                    #     outputs=[del_out, ds_status, file_table, preview_gallery, ds_stats_md],
                    # )

                    # ----- Build/Update output structure -----
                    def _build_and_refresh(ds, reps, res):
                        msg = cb_build_output_single(ds, int(reps), res_str=res)
                        _vis, status, table_upd, gal_upd, stats_upd = _cb_select_ds(ds)
                        return msg, status, table_upd, gal_upd, stats_upd

                    # (wiring moved below action buttons)
                    # build_btn.click(
                    #     _build_and_refresh,
                    #     inputs=[ds_dropdown, reps_slider, res_txt],
                    #     outputs=[build_out, ds_status, file_table, preview_gallery, ds_stats_md],
                    # )

                    # ---- External dataset sources management ----
                    def _fmt_src(rows):
                        return {"headers": ["Path"], "data": rows}

                    def _refresh_src_table():
                        return _fmt_src([[str(p)] for p in _ds_load_src()])

                    def _add_src(path):
                        if not path:
                            return _refresh_src_table(), "Enter path", _refresh_src_dropdown()
                        _ds_add_src(path)
                        return _refresh_src_table(), f"Added {path}", _refresh_src_dropdown()

                    def _extract_rows_any(data):
                        if data is None:
                            return []
                        if isinstance(data, dict):
                            return data.get("data", [])
                        try:
                            import pandas as pd
                            if isinstance(data, pd.DataFrame):
                                return data.values.tolist()
                        except Exception:
                            pass
                        return data if isinstance(data, list) else []

                    def _del_src(df_payload):
                        rows = _extract_rows_any(df_payload)
                        if not rows:
                            return _refresh_src_table(), "Select a row", _refresh_src_dropdown()
                        for row in rows:
                            if row and row[0]:
                                _ds_remove_src(row[0])
                        return _refresh_src_table(), "Removed", _refresh_src_dropdown()

                    def _refresh_src_dropdown():
                        return gr.update(choices=[str(p) for p in _ds_load_src()])

                    add_src_btn.click(_add_src, inputs=new_src_path, outputs=[src_list, ds_status, src_select])

                    del_src_btn.click(_del_src, inputs=src_list, outputs=[src_list, ds_status, src_select])

                    def _process_source(src_path_str: str):
                        from pathlib import Path
                        if not src_path_str:
                            return _fmt_src([[str(p)] for p in _ds_load_src()]), "Select source first"

                        src_path = Path(src_path_str)
                        if not src_path.exists():
                            return _fmt_src([[str(p)] for p in _ds_load_src()]), f"Path not found: {src_path}"

                        datasets = [d.name for d in src_path.iterdir() if d.is_dir()]
                        if not datasets:
                            return _fmt_src([[str(p)] for p in _ds_load_src()]), "No datasets in source"

                        success_cnt = 0
                        for name in datasets:
                            try:
                                cb_build_output_single(name)
                                generate_presets_for_dataset(name)
                                success_cnt += 1
                            except Exception:
                                pass

                        return _fmt_src([[str(p)] for p in _ds_load_src()]), f"Processed {success_cnt} / {len(datasets)} datasets"

                    proc_src_btn.click(_process_source, inputs=src_select, outputs=[src_list, ds_status])

                    # init table contents
                    src_list.value = _fmt_src([[str(p)] for p in _ds_load_src()])
                    src_select.value = None
                    src_select.choices = [str(p) for p in _ds_load_src()]

                    # ---- Action buttons (bottom of adjustments column) ----
                    gr.Markdown("---")
                    with gr.Row():
                        build_btn = gr.Button("Build / Update output structure", variant="primary", scale=2)
                    with gr.Row():
                        scan_btn = gr.Button("Scan duplicates / corrupt", scale=1)
                        del_btn = gr.Button("Delete ALL images", variant="stop", scale=1)

                    # ----- Wiring para los nuevos botones -----
                    scan_btn.click(_scan_ds, inputs=ds_dropdown, outputs=scan_out)

                    del_btn.click(lambda: None, [], [], js="() => confirm('Delete ALL images in this dataset folder?')").then(
                        _delete_and_refresh,
                        inputs=ds_dropdown,
                        outputs=[del_out, ds_status, file_table, preview_gallery, ds_stats_md],
                    )

                    build_btn.click(
                        _build_and_refresh,
                        inputs=[ds_dropdown, reps_slider, res_txt],
                        outputs=[build_out, ds_status, file_table, preview_gallery, ds_stats_md],
                    )

                with gr.TabItem("Bulk Prepare"):
                    gr.Markdown("### Bulk prepare multiple datasets")

                    def _bulk_build_table():
                        names = _list_dataset_raw()
                        rows = [[n, "", 30, ""] for n in names]
                        return {"headers": ["Dataset", "Trigger", "Reps", "Resolution"], "data": rows}

                    bulk_table = gr.Dataframe(
                        headers=["Dataset", "Trigger", "Reps", "Resolution"],
                        interactive=True,
                        row_count=(5, "dynamic"),
                        max_height=400,
                    )
                    bulk_table.value = _bulk_build_table()

                    with gr.Row():
                        refresh_bulk_btn = gr.Button("↻ Refresh list", scale=1)
                        run_bulk_btn = gr.Button("Run Bulk Prepare", variant="primary", scale=4)

                    cap_overwrite_bulk = gr.Checkbox(label="Overwrite existing captions", value=True)
                    cap_max_bulk = gr.Slider(label="Max tokens", minimum=32, maximum=512, step=32, value=128)

                    # Live log and progress
                    bulk_log_box = gr.Textbox(label="Log", lines=16, interactive=False)
                    bulk_prog_md = gr.Markdown()

                    refresh_bulk_btn.click(lambda: _bulk_build_table(), inputs=None, outputs=bulk_table)

                    def _extract_rows_any(data):
                        if data is None:
                            return []
                        if isinstance(data, dict):
                            return data.get("data", [])
                        try:
                            import pandas as pd
                            if isinstance(data, pd.DataFrame):
                                return data.values.tolist()
                        except Exception:
                            pass
                        return data if isinstance(data, list) else []

                    def _bulk_prepare_stream(rows_payload, overwrite, max_tok):
                        """Generator that prepares datasets sequentially and streams log & progress."""
                        rows = _extract_rows_any(rows_payload)
                        rows = [r for r in rows if r and r[0]]
                        total = len(rows)
                        if total == 0:
                            yield ("No datasets specified.", "")
                            return

                        log_lines = []
                        completed = 0
                        for row in rows:
                            ds = str(row[0]).strip()
                            trig = str(row[1]).strip() if len(row) > 1 else ""
                            reps = int(row[2]) if len(row) > 2 and str(row[2]).strip() else 30
                            res = str(row[3]).strip() if len(row) > 3 else ""

                            cap_msg = ""
                            if trig:
                                try:
                                    from .captioner import rename_and_caption_dataset
                                    n_caps = rename_and_caption_dataset(ds, trig, overwrite=overwrite, max_new_tokens=int(max_tok))
                                    cap_msg = f" · {n_caps} caption(s)"
                                except Exception as e:
                                    cap_msg = f" · Captioning error: {e}"

                            # Build / update output structure after renaming so only final files are copied
                            build_msg = cb_build_output_single(ds, reps, res_str=None)  # create structure & copy files

                            crop_msg = ""
                            if res:
                                try:
                                    special_folder = OUTPUT_DIR / ds / "img" / f"{repeats}_{ds} person"
                                    processed = _crop_images_in_folder(special_folder, res)
                                    crop_msg = f" · Cropped {processed}"
                                except Exception as e:
                                    crop_msg = f" · Crop error: {e}"

                            log_lines.append(f"{ds}: {build_msg}{cap_msg}{crop_msg}")
                            completed += 1
                            progress_str = f"{completed}/{total} done"
                            yield ("\n".join(log_lines[-400:]), progress_str)

                    run_bulk_btn.click(
                        _bulk_prepare_stream,
                        inputs=[bulk_table, cap_overwrite_bulk, cap_max_bulk],
                        outputs=[bulk_log_box, bulk_prog_md],
                    )

        with gr.Tab("Configuration"):
            gr.Markdown("### Generate training presets")
            ds_select_cfg = gr.Dropdown(label="Select dataset", choices=_list_dataset_raw())
            with gr.Row():
                gen_btn = gr.Button("Generate presets", variant="primary")
                gen_all_btn = gr.Button("Generate ALL presets")
                refresh_btn_cfg = gr.Button("Refresh list")
            gen_status = gr.Markdown()

            with gr.Accordion("Override presets", open=False):
                with gr.Row():
                    ds_edit = gr.Dropdown(label="Select dataset", choices=_list_dataset_raw(), scale=8)
                    refresh_override = gr.Button("Refresh editors", scale=2)

                # Carousel Tabs
                with gr.Tabs():
                    with gr.TabItem("FluxCheckpoint"):
                        code_flux = gr.Textbox(label="FluxCheckpoint TOML", lines=20)
                        save_flux = gr.Button("Save Flux")

                    with gr.TabItem("FluxLORA"):
                        code_lora = gr.Textbox(label="FluxLORA TOML", lines=20)
                        save_lora = gr.Button("Save FluxLORA")

                    with gr.TabItem("SDXLNude"):
                        code_nude = gr.Textbox(label="SDXLNude TOML", lines=20)
                        save_nude = gr.Button("Save Nude")

                save_status = gr.Markdown()

                def _paths(dataset):
                    return (
                        BATCH_CONFIG_DIR / "Flux" / f"{dataset}.toml",
                        BATCH_CONFIG_DIR / "FluxLORA" / f"{dataset}.toml",
                        BATCH_CONFIG_DIR / "Nude" / f"{dataset}.toml",
                    )

                def _load_tomls(dataset):
                    if not dataset:
                        return "", "", ""
                    p_flux, p_lora, p_nude = _paths(dataset)
                    read = lambda p: (p.read_text() if p.exists() else "# file not found")
                    return (
                        gr.update(value=read(p_flux)),
                        gr.update(value=read(p_lora)),
                        gr.update(value=read(p_nude)),
                    )

                ds_edit.change(_load_tomls, inputs=ds_edit, outputs=[code_flux, code_lora, code_nude])

                def _save(profile, dataset, content):
                    if not dataset:
                        return "Select a dataset"
                    p = _paths(dataset)[profile]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content)
                    return f"Saved {p}"

                save_flux.click(lambda d, c: _save(0, d, c), inputs=[ds_edit, code_flux], outputs=save_status)
                save_lora.click(lambda d, c: _save(1, d, c), inputs=[ds_edit, code_lora], outputs=save_status)
                save_nude.click(lambda d, c: _save(2, d, c), inputs=[ds_edit, code_nude], outputs=save_status)

                # Refresh editors button callback
                refresh_override.click(lambda d: _load_tomls(d), inputs=ds_edit, outputs=[code_flux, code_lora, code_nude])

            gen_btn.click(_cb_gen_presets, inputs=ds_select_cfg, outputs=[gen_status, ds_edit])

            def _cb_gen_all():
                names = _list_dataset_raw()
                if not names:
                    return "No datasets found", gr.update()
                for n in names:
                    try:
                        populate_output_structure_single(n)
                    except Exception:
                        pass
                    generate_presets_for_dataset(n)
                return f"Presets generated for {len(names)} datasets", _list_dataset_choices()

            gen_all_btn.click(_cb_gen_all, inputs=None, outputs=[gen_status, ds_edit])

            # refresh button just updates dropdown
            refresh_btn_cfg.click(lambda: _list_dataset_choices(), inputs=[], outputs=ds_select_cfg)
            refresh_btn_cfg.click(lambda: _list_dataset_choices(), inputs=[], outputs=ds_edit)

            # --- Aplicar resolución del dataset ---
            apply_res_btn = gr.Button("Apply dataset resolution", variant="secondary")

            def _infer_dataset_resolution(dataset: str) -> str | None:
                """Return "WxH" from images already in output/<dataset> or fallback to input."""
                # 1) Try output folder (may contain cropped/resized images)
                out_base = OUTPUT_DIR / dataset
                if out_base.exists():
                    # search recursively for first png/jpg
                    for ext in SUPPORTED_IMAGE_EXTS:
                        imgs = list(out_base.rglob(f"*.{ext}"))
                        if imgs:
                            try:
                                from PIL import Image
                                with Image.open(imgs[0]) as im:
                                    w, h = im.size
                                    if w > 0 and h > 0:
                                        return f"{w},{h}"
                            except Exception:
                                pass
                # 2) Fallback to original input dataset
                folder = _resolve_dataset_path(dataset)
                if folder is None:
                    return None
                for ext in SUPPORTED_IMAGE_EXTS:
                    for p in folder.glob(f"*.{ext}"):
                        try:
                            from PIL import Image
                            with Image.open(p) as im:
                                w, h = im.size
                                if w > 0 and h > 0:
                                    return f"{w},{h}"
                        except Exception:
                            pass
                return None

            def _apply_dataset_res(dataset, txt_flux, txt_lora, txt_nude):
                import toml as _toml
                if not dataset:
                    return txt_flux, txt_lora, txt_nude, "Select dataset first"
                res = _infer_dataset_resolution(dataset)
                if res is None:
                    return txt_flux, txt_lora, txt_nude, "Error: You must build the output structure first."

                def _patch(toml_str: str) -> str:
                    try:
                        cfg = _toml.loads(toml_str)
                    except Exception:
                        return toml_str  # cannot parse
                    cfg["resolution"] = res  # e.g. "1024,1024"
                    return _toml.dumps(cfg)

                patched_flux = _patch(txt_flux)
                patched_lora = _patch(txt_lora)
                patched_nude = _patch(txt_nude)
                return patched_flux, patched_lora, patched_nude, f"Resolution set to {res}"

            apply_res_btn.click(
                _apply_dataset_res,
                inputs=[ds_edit, code_flux, code_lora, code_nude],
                outputs=[code_flux, code_lora, code_nude, save_status],
            )

        with gr.Tab("Training"):
            gr.Markdown("## 🚀 Model Training")
            
            # === CONFIGURATION SECTION ===
            with gr.Accordion("⚙️ Configuration", open=True):
                with gr.Row():
                    # Training Mode
                    with gr.Column(scale=1):
                        training_mode_toggle = gr.Radio(
                            ["Single Mode", "Batch Mode"], 
                            label="🎯 Training Mode", 
                            value="Single Mode",
                            info="Single: Train one dataset. Batch: Train multiple datasets in sequence."
                        )
                    
                    # Training Profile
                    with gr.Column(scale=1):
                        prof = gr.Dropdown(
                            ["Flux", "FluxLORA", "Nude"], 
                            label="🔧 Training Profile", 
                            value="Flux",
                            info="Choose the training architecture"
                        )
                    
                    # Action Buttons
                    with gr.Column(scale=1):
                        with gr.Row():
                            run_btn = gr.Button("🚀 Start Training", variant="primary", scale=2)
                            cancel_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)

            # === DATASET SELECTION SECTION ===
            with gr.Accordion("📂 Dataset Selection", open=True):
                # Single Mode UI
                single_mode_group = gr.Group(visible=True)
                with single_mode_group:
                    with gr.Row():
                        ds_train = gr.Dropdown(
                            label="Select Dataset", 
                            choices=_list_dataset_in_profile("Flux"),
                            scale=4
                        )
                        refresh_btn_train = gr.Button("↻ Refresh", variant="secondary", scale=1)
                
                # Batch Mode UI
                batch_mode_group = gr.Group(visible=False)
                with batch_mode_group:
                    batch_datasets = gr.CheckboxGroup(
                        choices=_list_dataset_in_profile("Flux"),
                        label="Available Datasets",
                        info="Select multiple datasets to train in sequence"
                    )
                    refresh_batch_btn = gr.Button("↻ Refresh Dataset List", variant="secondary")

            # === DATASET INFO SECTION ===
            with gr.Accordion("📊 Dataset Info", open=True):
                with gr.Row():
                    # Left Column - Dataset Information
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Dataset Information")
                        dataset_info_md = gr.Markdown("**📂 Dataset:** _None selected_\n\n📝 **Info:** Select a dataset to see information")

                    # Center Column - Preview Gallery
                    with gr.Column(scale=2):
                        gr.Markdown("### 🖼️ Preview Gallery")
                        dataset_preview_gallery = gr.Gallery(
                            label="Sample Images",
                            columns=2,
                            rows=2,
                            height=300,
                            visible=False
                        )

                    # Right Column - Training Estimation (now hidden)
                    with gr.Column(scale=1, visible=False):
                        estimation_md = gr.Markdown(visible=False)

            # Output path information (legacy – now hidden)
            path_md = gr.Markdown(visible=False)

            # === MONITORING SECTION ===
            gr.Markdown("---")
            gr.Markdown("## 📈 Training Monitor")
            
            # TOP: Queue Status Bar
            with gr.Row():
                queue_status = gr.Markdown("**Waiting for job to load statistics...**")
            # Removed the manual refresh button
            
            with gr.Row():
                # LEFT: Current Job & Summary
                with gr.Column(scale=2):
                    with gr.Accordion("🔥 Active Training", open=True):
                        current_job_status = gr.Markdown("**No active training job**")
                    # Upcoming jobs list
                    with gr.Accordion("📅 Upcoming Jobs", open=True):
                        upcoming_jobs_md = gr.Markdown("_No upcoming jobs_")

                # RIGHT: Visual Outputs
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("🖼️ Generated Samples"):
                            gallery = gr.Gallery(
                                label="Generated Samples", 
                                columns=3, 
                                height=400,
                                show_label=False
                            )
                        
                        with gr.TabItem("📝 Training Log"):
                            log_box = gr.Textbox(
                                label="Training Log", 
                                lines=16, 
                                interactive=False,
                                show_label=False,
                                placeholder="Training logs will appear here..."
                            )

            # Toggle visibility based on training mode
            def _toggle_training_mode(mode):
                if mode == "Single Mode":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            training_mode_toggle.change(
                _toggle_training_mode,
                inputs=[training_mode_toggle],
                outputs=[single_mode_group, batch_mode_group]
            )

            # Update info when training mode changes
            def _update_info_on_mode_change(mode, single_dataset, batch_datasets_list, profile):
                """Update dataset info when switching between single and batch mode."""
                if mode == "Single Mode":
                    return _update_dataset_info(single_dataset, profile)
                else:
                    return _update_batch_info(batch_datasets_list, profile)

            training_mode_toggle.change(
                _update_info_on_mode_change,
                inputs=[training_mode_toggle, ds_train, batch_datasets, prof],
                outputs=[dataset_info_md, dataset_preview_gallery, estimation_md]
            )

            # Update dataset lists when profile changes
            def _update_dataset_lists(profile):
                datasets = _list_dataset_in_profile(profile)
                return (
                    gr.update(choices=datasets, value=None),
                    gr.update(choices=datasets, value=[])
                )

            prof.change(
                _update_dataset_lists,
                inputs=[prof],
                outputs=[ds_train, batch_datasets]
            )

            # Dataset information and estimation callbacks
            def _update_dataset_info(dataset_name, profile):
                """Update dataset information panel."""
                if not dataset_name:
                    return (
                        "**📂 Dataset:** _None selected_\n\n📝 **Info:** Select a dataset to see information",
                        gr.update(visible=False),
                        "**⏱️ Training Estimation**\n\nSelect a dataset to see time estimate"
                    )
                
                # Get dataset info
                dataset_info = _get_dataset_info(dataset_name)
                
                if not dataset_info.get("exists", False):
                    error_msg = dataset_info.get("error", "Dataset not found")
                    info_md = f"**📂 Dataset:** {dataset_name}\n\n❌ **Error:** {error_msg}"
                    est_md = f"**⏱️ Training Estimation**\n\n❌ {error_msg}"
                    return info_md, gr.update(visible=False), est_md
                
                # Format dataset information (line by line)
                info_lines = [
                    f"**📂 Dataset:** {dataset_name}",
                    f"🖼️ **Images:** {dataset_info['num_images']}",
                    f"📝 **Captions:** {dataset_info['num_captions']}",
                    f"📐 **Resolution:** {dataset_info['avg_resolution']}",
                    f"💾 **Size:** {dataset_info['total_size']}",
                    f"📅 **Modified:** {dataset_info['last_modified']}",
                    f"📁 **Path:** `{dataset_info['path']}`"
                ]
                
                info_md = "\n\n".join(info_lines)
                
                # Update preview gallery
                sample_images = dataset_info.get("sample_images", [])
                gallery_update = gr.update(
                    value=sample_images,
                    visible=len(sample_images) > 0
                )
                
                # Get training estimation
                estimation = _estimate_training_time(dataset_name, profile)
                
                if "error" in estimation:
                    est_md = f"**⏱️ Training Estimation**\n\n❌ {estimation['error']}"
                else:
                    est_lines = [
                        f"**⏱️ Training Estimation: {dataset_name}**",
                        "",
                        f"⏰ **Estimated Time:** {estimation['estimated_time']}",
                        f"🔄 **Total Steps:** {estimation['total_steps']:,}",
                        f"🎮 **GPU:** {estimation['gpu_name']}",
                        f"⚡ **Power Consumption:** {estimation['power_consumption_kwh']} kWh",
                        "",
                        "**Parameters:**",
                        f"• **Epochs:** {estimation['parameters']['epochs']}",
                        f"• **Batch Size:** {estimation['parameters']['batch_size']}",
                        f"• **Repeats:** {estimation['parameters']['repeats']}",
                        "",
                        "*Note: Times are estimates based on your GPU profile*"
                    ]
                    est_md = "\n".join(est_lines)
                
                return info_md, gallery_update, est_md

            def _update_batch_info(selected_datasets, profile):
                """Update information for batch mode."""
                if not selected_datasets:
                    return (
                        "**📂 Datasets:** _None selected_\n\n📝 **Info:** Select datasets to see information",
                        gr.update(visible=False),
                        "**⏱️ Batch Training Estimation**\n\nSelect datasets to see time estimate"
                    )
                
                # Collect info for all selected datasets
                total_images = 0
                total_captions = 0
                total_size_bytes = 0
                total_estimated_minutes = 0
                total_power_consumption = 0
                dataset_details = []
                all_sample_images = []
                gpu_name = ""
                
                for dataset_name in selected_datasets:
                    dataset_info = _get_dataset_info(dataset_name)
                    if dataset_info.get("exists", False):
                        total_images += dataset_info.get("num_images", 0)
                        total_captions += dataset_info.get("num_captions", 0)
                        
                        # Parse size back to bytes for totaling
                        size_str = dataset_info.get("total_size", "0 B")
                        try:
                            if " GB" in size_str:
                                size_val = float(size_str.replace(" GB", ""))
                                total_size_bytes += size_val * 1024 * 1024 * 1024
                            elif " MB" in size_str:
                                size_val = float(size_str.replace(" MB", ""))
                                total_size_bytes += size_val * 1024 * 1024
                            elif " KB" in size_str:
                                size_val = float(size_str.replace(" KB", ""))
                                total_size_bytes += size_val * 1024
                            else:
                                size_val = float(size_str.replace(" B", ""))
                                total_size_bytes += size_val
                        except:
                            pass
                        
                        # Get estimation for this dataset
                        estimation = _estimate_training_time(dataset_name, profile)
                        if "estimated_minutes" in estimation:
                            total_estimated_minutes += estimation["estimated_minutes"]
                        if "power_consumption_kwh" in estimation:
                            total_power_consumption += estimation["power_consumption_kwh"]
                        if "gpu_name" in estimation and not gpu_name:
                            gpu_name = estimation["gpu_name"]
                        
                        # Add to details
                        dataset_details.append(
                            f"• **{dataset_name}**: {dataset_info['num_images']} images, {dataset_info['avg_resolution']}"
                        )
                        
                        # Add sample images (max 2 per dataset)
                        sample_images = dataset_info.get("sample_images", [])
                        all_sample_images.extend(sample_images[:2])
                
                # Format total size
                if total_size_bytes < 1024:
                    total_size_str = f"{total_size_bytes:.0f} B"
                elif total_size_bytes < 1024 * 1024:
                    total_size_str = f"{total_size_bytes / 1024:.1f} KB"
                elif total_size_bytes < 1024 * 1024 * 1024:
                    total_size_str = f"{total_size_bytes / (1024 * 1024):.1f} MB"
                else:
                    total_size_str = f"{total_size_bytes / (1024 * 1024 * 1024):.1f} GB"
                
                # Format batch information (line by line)
                info_lines = [
                    f"**📂 Datasets:** {len(selected_datasets)} selected",
                    f"🖼️ **Total Images:** {total_images:,}",
                    f"📝 **Total Captions:** {total_captions:,}",
                    f"💾 **Total Size:** {total_size_str}",
                    "",
                    "**📋 Selected Datasets:**"
                ] + dataset_details
                
                info_md = "\n\n".join(info_lines[:4]) + "\n\n" + info_lines[4] + "\n" + "\n".join(info_lines[5:])
                
                # Update preview gallery
                gallery_update = gr.update(
                    value=all_sample_images[:8],  # Max 8 sample images
                    visible=len(all_sample_images) > 0
                )
                
                # Format total estimation
                if total_estimated_minutes < 60:
                    total_time_str = f"{int(total_estimated_minutes)} min"
                elif total_estimated_minutes < 1440:  # 24 hours
                    hours = int(total_estimated_minutes // 60)
                    mins = int(total_estimated_minutes % 60)
                    total_time_str = f"{hours}h {mins}m"
                else:
                    days = int(total_estimated_minutes // 1440)
                    hours = int((total_estimated_minutes % 1440) // 60)
                    total_time_str = f"{days}d {hours}h"
                
                est_lines = [
                    f"**⏱️ Batch Training Estimation ({len(selected_datasets)} datasets)**",
                    "",
                    f"⏰ **Total Estimated Time:** {total_time_str}",
                    f"📊 **Datasets:** {len(selected_datasets)}",
                    f"🖼️ **Total Images:** {total_images:,}",
                    f"🎮 **GPU:** {gpu_name}",
                    f"⚡ **Total Power Consumption:** {total_power_consumption:.2f} kWh",
                    "",
                    "*Note: Datasets will be trained sequentially*",
                    "*Times are estimates based on your GPU profile*"
                ]
                
                est_md = "\n".join(est_lines)
                
                return info_md, gallery_update, est_md

            # Update info when dataset selection changes (Single Mode)
            ds_train.change(
                _update_dataset_info,
                inputs=[ds_train, prof],
                outputs=[dataset_info_md, dataset_preview_gallery, estimation_md]
            )

            # Update info when batch dataset selection changes (Batch Mode)
            batch_datasets.change(
                _update_batch_info,
                inputs=[batch_datasets, prof],
                outputs=[dataset_info_md, dataset_preview_gallery, estimation_md]
            )

            # Update info when profile changes (for currently selected dataset)
            def _update_info_on_profile_change(profile, current_dataset):
                return _update_dataset_info(current_dataset, profile)

            prof.change(
                _update_info_on_profile_change,
                inputs=[prof, ds_train],
                outputs=[dataset_info_md, dataset_preview_gallery, estimation_md]
            )

            # Refresh dataset lists
            refresh_btn_train.click(
                lambda profile: gr.update(choices=_list_dataset_in_profile(profile), value=None),
                inputs=[prof],
                outputs=[ds_train]
            )
            
            refresh_batch_btn.click(
                lambda profile: gr.update(choices=_list_dataset_in_profile(profile), value=[]),
                inputs=[prof],
                outputs=[batch_datasets]
            )

            # Training execution logic
            def _execute_training(mode, profile, single_dataset, batch_datasets_list):
                if mode == "Single Mode":
                    return cb_start_training(profile, single_dataset)
                else:
                    return cb_start_batch_training(profile, batch_datasets_list)

            run_btn.click(
                _execute_training,
                inputs=[training_mode_toggle, prof, ds_train, batch_datasets],
                outputs=[path_md, log_box, gallery]
            )
            
            cancel_btn.click(cb_cancel_training, inputs=None, outputs=log_box)

            # Enhanced polling functions for improved metrics
            def _poll_enhanced_metrics():
                """Poll enhanced metrics and present them in a concise format."""
                log, pics = JOB_MANAGER.get_live_output()
                job = JOB_MANAGER.get_current_job()
                
                # Queue information
                all_jobs = JOB_MANAGER.list_jobs()
                running_jobs = [j for j in all_jobs if j.status.name == "RUNNING"]
                pending_jobs = [j for j in all_jobs if j.status.name == "PENDING"]
                completed_jobs = [j for j in all_jobs if j.status.name in ["DONE", "FAILED", "CANCELED"]]
                
                # Build queue summary as a 4-column markdown table for clear grid layout
                running_cnt = len(running_jobs)
                pending_cnt = len(pending_jobs)
                completed_cnt = len(completed_jobs)
                total_cnt = len(all_jobs)

                queue_text = (
                    "| 🔥 Running | ⏳ Pending | ✅ Completed | 📦 Total |\n"
                    "| --- | --- | --- | --- |\n"
                    f"| {running_cnt} | {pending_cnt} | {completed_cnt} | {total_cnt} |"
                )
                
                # If queue is empty, reset widgets to initial waiting state
                if running_cnt == 0 and pending_cnt == 0:
                    return (
                        "**Waiting for job to load statistics...**",  # queue_status
                        "**No active training job**",                  # current_job_status
                        "_No upcoming jobs_",                         # upcoming jobs
                        "",                                           # log box empty
                        []                                             # gallery empty
                    )
                
                if job:
                    # Build status lines – each piece of info on its own line
                    job_status_lines = [
                        f"**🎯 Job:** `{job.id}`",
                        f"**📂 Dataset:** {job.dataset}",
                        f"**⚙️ Profile:** {job.profile}",
                        f"**📊 Status:** {job.status.name}"
                    ]
                    # Progress percentage
                    if job.percent > 0:
                        job_status_lines.append(f"**📈 Progress:** {job.percent:.1f}%")
                    # ETA
                    if job.eta:
                        job_status_lines.append(f"**⏰ ETA:** {job.eta}")
                    # Step info
                    if job.current_step > 0 and job.total_steps > 0:
                        job_status_lines.append(f"**🔄 Step:** {job.current_step:,}/{job.total_steps:,}")
                    elif job.current_step > 0:
                        job_status_lines.append(f"**🔄 Step:** {job.current_step:,}")
                    # Epoch (max epochs) – read from job's TOML once and cache it
                    try:
                        if hasattr(job, "_max_epochs"):
                            max_epochs = job._max_epochs
                        else:
                            import toml as _toml
                            cfg_tmp = _toml.load(job.toml_path)
                            max_epochs = cfg_tmp.get("max_train_epochs", "-")
                            job._max_epochs = max_epochs  # cache for later
                        job_status_lines.append(f"**📅 Epoch:** {max_epochs}")
                    except Exception:
                        job_status_lines.append("**📅 Epoch:** -")
                    
                    # Output directory
                    job_status_lines.append(f"**📁 Output dir:** `{job.run_dir}`")
                    
                    # Join with <br> so each item is rendered on its own line in markdown
                    job_status_text = "<br>".join(job_status_lines)
                    
                    # Build upcoming jobs markdown (next 10 pending)
                    upcoming_list = pending_jobs[:10]
                    if upcoming_list:
                        md_lines = ["| Job | Dataset | Profile |", "| --- | --- | --- |"]
                        for j in upcoming_list:
                            md_lines.append(f"| `{j.id}` | {j.dataset} | {j.profile} |")
                        upcoming_md = "\n".join(md_lines)
                    else:
                        upcoming_md = "_No upcoming jobs_"
                    
                    return (
                        queue_text,
                        job_status_text,
                        upcoming_md,
                        log,
                        pics,
                    )
                else:
                    no_job_text = "**🚫 No Active Training**\nNo running jobs"
                    # Upcoming jobs markdown (first 10 pending)
                    upcoming_list = pending_jobs[:10]
                    if upcoming_list:
                        md_lines = ["| Job | Dataset | Profile |", "| --- | --- | --- |"]
                        for j in upcoming_list:
                            md_lines.append(f"| `{j.id}` | {j.dataset} | {j.profile} |")
                        upcoming_md = "\n".join(md_lines)
                    else:
                        upcoming_md = "_No upcoming jobs_"
                    return (
                        queue_text,
                        no_job_text,
                        upcoming_md,
                        log,
                        pics,
                    )

            # Timer to poll enhanced metrics every 3 seconds
            gr.Timer(3.0).tick(
                _poll_enhanced_metrics,
                None,
                [
                    queue_status, current_job_status, upcoming_jobs_md, log_box, gallery
                ]
            )

        # ---------------- QuickTrain Tab -----------------
        with gr.Tab("QuickTrain"):
            gr.Markdown("""### One-click training

Specify your dataset folder and choose the mode. The system will import the dataset, create the *output/* structure, generate the presets and enqueue the training automatically.""")

            with gr.Row():
                ds_dropdown_qt = gr.Dropdown(label="Select existing dataset", choices=_list_dataset_raw(), scale=6)
                refresh_qt = gr.Button("↻", scale=1)

            ds_path_box = gr.Textbox(label="…or provide external folder path (absolute)")
            remote_chk2 = gr.Checkbox(label="Store output in remote path (AUTO_REMOTE_BASE)" , value=False)
            mode_qt = gr.Dropdown(["Flux","FluxLORA","Nude"], label="Training mode", value="Flux")

            with gr.Accordion("Advanced parameters", open=False):
                epochs_box = gr.Number(label="Max epochs (blank = preset)", precision=0)
                batch_box = gr.Number(label="Train batch size", precision=0)
                lr_box = gr.Number(label="Learning rate", precision=5)

            qt_btn = gr.Button("Start QuickTrain", variant="primary")
            qt_out = gr.Markdown()

            refresh_qt.click(lambda: gr.update(choices=_list_dataset_raw()), inputs=None, outputs=ds_dropdown_qt)

            # Core helper (extracted from previous implementation)
            def _qt_core(path_str: str, use_remote: bool, mode_sel: str, epochs=None, batch=None, lr=None):
                from pathlib import Path
                import os, shutil, toml

                if not path_str:
                    return "❌ Provide dataset path"
                src = Path(path_str)
                if not src.exists() or not src.is_dir():
                    return f"❌ Path not found: {src}"

                dataset_name = src.name

                dest = INPUT_DIR / dataset_name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                    except Exception as e:
                        return f"❌ Copy error: {e}"

                if use_remote:
                    os.environ["AUTO_REMOTE_DIRECT"] = "1"
                else:
                    os.environ.pop("AUTO_REMOTE_DIRECT", None)

                try:
                    populate_output_structure_single(dataset_name)
                    generate_presets_for_dataset(dataset_name)
                except Exception as e:
                    return f"❌ Prep error: {e}"

                preset_dir = BATCH_CONFIG_DIR / ("Flux" if mode_sel=="Flux" else ("FluxLORA" if mode_sel=="FluxLORA" else "Nude"))
                cfg_path = preset_dir / f"{dataset_name}.toml"
                if not cfg_path.exists():
                    return f"❌ Preset not found: {cfg_path}"

                orig_cfg = toml.load(cfg_path)
                # apply overrides
                if epochs:
                    orig_cfg["max_train_epochs"] = int(epochs)
                if batch:
                    orig_cfg["train_batch_size"] = int(batch)
                if lr:
                    orig_cfg["learning_rate"] = float(lr)

                run_dir = compute_run_dir(dataset_name, mode_sel)
                run_dir.mkdir(parents=True, exist_ok=True)

                patched_cfg_path = run_dir/"config.toml"
                with patched_cfg_path.open("w",encoding="utf-8") as f:
                    toml.dump(orig_cfg,f)

                job = Job(dataset_name, mode_sel, patched_cfg_path, run_dir)
                JOB_MANAGER.enqueue(job)

                return f"✅ Job {job.id} queued. Output → `{run_dir}`"

            def _wrapper_qt(sel_name,path_str,use_remote,mode_sel,ep,batchp,lrp):
                # prefer explicit path, else use dataset name
                chosen_path = path_str.strip() if path_str else ""
                if not chosen_path and sel_name:
                    # resolve to input/ or external source
                    from pathlib import Path
                    p_in = INPUT_DIR / sel_name
                    if p_in.exists():
                        chosen_path = str(p_in)
                    else:
                        extp = _ds_find_path(sel_name)
                        if extp:
                            chosen_path = str(extp)
                return _qt_core(chosen_path, use_remote, mode_sel, ep or None, batchp or None, lrp or None)

            qt_btn.click(_wrapper_qt, inputs=[ds_dropdown_qt, ds_path_box, remote_chk2, mode_qt, epochs_box, batch_box, lr_box], outputs=qt_out)

        # ---------------- Experiments Tab -----------------
        with gr.Tab("Experiments"):
            gr.Markdown("""### Experiments (A/B testing)

Create *many* training runs for the **same dataset** by changing hyper-parameters and/or profile. This helps you compare which settings give the best result.

**How to use**
1. Pick the *Dataset* and *Base profile* (Flux / FluxLORA / Nude). The parameter table loads the current TOML.
2. In the **Variations** column type the values you want to test for any argument:
   • Single value → override default.
   • Comma list `a,b,c` → one run per value.
   • Numeric range `start:stop:step` → e.g. `1:5:1` 
     (generates 1,2,3,4,5).
3. Click **Launch Experiment**. The Cartesian product of all variations is enqueued (can be dozens of runs!).
4. Track progress in *Queue*. When every run is *done* use **Refresh Experiments** and **Compare variants** to see a summary.

Tip: leave the *Variations* cell empty to keep that argument unchanged.
""")

            with gr.Row():
                ds_exp = gr.Dropdown(label="Dataset", choices=_list_dataset_raw(), scale=6)
                refresh_ds_exp = gr.Button("↻", scale=1)

            prof_exp = gr.Dropdown(["Flux", "FluxLORA", "Nude"], label="Base profile", value="Flux")

            # Refresh dataset list callback
            refresh_ds_exp.click(lambda: _list_dataset_choices(), inputs=None, outputs=ds_exp)

            var_table = gr.Dataframe(
                headers=["Profile", "Epochs", "LR"],
                row_count=(3, "dynamic"),
                interactive=True,
            )

            launch_exp_btn = gr.Button("Launch Experiment", variant="primary")
            exp_msg = gr.Markdown()

            # experiment list
            list_exp_btn = gr.Button("Refresh Experiments")
            with gr.Row():
                confirm_chk = gr.Checkbox(label="⚠️ I understand this will delete all experiments", value=False)
                clear_exp_btn = gr.Button("Clear experiments", variant="stop", interactive=False)
            exp_table = gr.Dataframe(headers=["ID", "Dataset", "#Runs", "Status"], interactive=False)

            sel_exp = gr.Dropdown(label="Select experiment (done)")
            compare_btn = gr.Button("Compare variants")
            compare_md = gr.Markdown()

            def _parse_variation(val_str, current_val):
                val_str = val_str.strip()
                if not val_str:
                    return [current_val]
                # range a:b:c
                if ":" in val_str:
                    parts = val_str.split(":")
                    if len(parts) in (2,3) and all(p.strip().replace('.','',1).replace('-','',1).isdigit() for p in parts):
                        start = float(parts[0]); stop = float(parts[1]); step = float(parts[2]) if len(parts)==3 else 1
                        seq = []
                        v=start
                        while (step>0 and v<=stop) or (step<0 and v>=stop):
                            seq.append(type(current_val)(v))
                            v += step
                        return seq
                # comma list
                items=[x.strip() for x in val_str.split(",") if x.strip()]
                if isinstance(current_val,(int,float)):
                    def cast(x):
                        try:
                            return type(current_val)(float(x))
                        except Exception:
                            return current_val
                    return [cast(x) for x in items]
                return items

            def _generate_variants(base_cfg: dict, rows):
                keys=[]; values_list=[]
                for arg, cur, var in rows:
                    if not arg:
                        continue
                    cur_val=base_cfg.get(arg,cur)
                    var_options=_parse_variation(str(var),cur_val)
                    if len(var_options)==1 and var_options[0]==cur_val:
                        continue
                    keys.append(arg)
                    values_list.append(var_options)
                all_variants=[]
                for combo in itertools.product(*values_list):
                    overrides=dict(zip(keys,combo))
                    all_variants.append(overrides)
                # Filtrar combinaciones que no cambian ningún valor respecto al base
                filtered=[]
                for ov in all_variants:
                    if any(base_cfg.get(k)!=v for k,v in ov.items()):
                        filtered.append(ov)
                if not filtered:
                    # si todas eran iguales al base preserva una variante vacía para no quedarse sin runs
                    filtered.append({})
                return filtered

            def _launch_experiment(dataset, base_profile, table_payload):
                """Genera variantes y las envía a la cola.  Muestra progreso en vivo."""

                # Paso 0: validaciones
                if not dataset:
                    yield "Select dataset first"
                    return

                # Mensaje inicial
                yield "⏳ Sending orders…"

                rows = _extract_rows_any(table_payload)

                # Cargar config base
                base_path = BATCH_CONFIG_DIR / base_profile / f"{dataset}.toml"
                try:
                    cfg_base = load_config(base_path)
                except FileNotFoundError:
                    yield f"Base TOML not found for {base_profile}"
                    return

                # Generar variaciones
                overrides_list = _generate_variants(cfg_base, rows)
                variants = [{"profile": base_profile, "overrides": ov} for ov in overrides_list]

                # Crear experimento y encolar jobs
                exp = create_experiment(dataset, variants)

                # Mensaje final
                yield f"✅ Experiment {exp.id} launched with {len(variants)} variants"

            launch_exp_btn.click(_launch_experiment, inputs=[ds_exp, prof_exp, var_table], outputs=exp_msg)

            def _list_experiments():
                from .experiments import EXP_DIR
                jobs = JOB_MANAGER.list_jobs()
                rows = []
                done_ids = []
                for p in EXP_DIR.glob("*.json"):
                    try:
                        data = json.loads(p.read_text())
                        exp_id = data["id"]
                        runs_total = len(data["runs"])
                        statuses = [j.status for j in jobs if j.experiment_id == exp_id]
                        if not statuses:
                            status = data.get("status", "planned")
                        elif any(s in (JobStatus.PENDING, JobStatus.RUNNING) for s in statuses):
                            status = "running"
                        elif all(s == JobStatus.DONE for s in statuses):
                            status = "done"
                        else:
                            status = "mixed"
                        rows.append([exp_id, data["dataset"], runs_total, status])
                        if status == "done":
                            done_ids.append(exp_id)
                    except Exception:
                        pass
                return rows, gr.update(choices=done_ids)

            list_exp_btn.click(lambda: _list_experiments(), inputs=None, outputs=[exp_table, sel_exp])

            # initial load
            rows, sel_upd = _list_experiments()
            exp_table.value = rows
            sel_exp.value = None
            sel_exp.choices = sel_upd["choices"] if isinstance(sel_upd, dict) else []

            # Compare button
            def _compare(exp_id):
                if not exp_id:
                    return "Select experiment"
                from .experiments import Experiment
                from .run_registry import _load as _rr_load  # load job registry
                exp = Experiment.load(exp_id)

                # Pre-load registry records for this experiment to map idx→job_id
                recs_exp = [rec for rec in _rr_load().values() if rec.get("experiment_id") == exp_id]

                def _job_for_idx(i: int) -> str:
                    """Return job_id corresponding to the variant index *i* (or '-')"""
                    tag = f"tmp_{exp_id}_{i}.toml"
                    for rec in recs_exp:
                        if tag in str(rec.get("toml_path", "")):
                            return rec.get("job_id", "-")
                    return "-"

                md_lines = [f"## Experiment {exp_id} – {exp.dataset}", ""]
                md_lines.append("### Variants")
                md_lines.append("| Job | Profile | Overrides |")
                md_lines.append("|-----|---------|-----------|")

                for idx, rs in enumerate(exp.runs):
                    prof = rs.get("profile", "Flux")
                    overrides = rs.get("overrides", {})
                    job_code = _job_for_idx(idx)
                    ov_str = "<br>".join(f"`{k}` = **{v}**" for k, v in overrides.items()) if overrides else "-"
                    md_lines.append(f"| {job_code} | {prof} | {ov_str} |")

                return "\n".join(md_lines)

            compare_btn.click(_compare, inputs=sel_exp, outputs=compare_md)

            # When dataset or profile changes, load param table
            def _load_param_table(*vals):
                dataset = vals[0] if len(vals) > 0 else None
                profile = vals[1] if len(vals) > 1 else "Flux"
                if not dataset:
                    return {"headers":["Argument","Current","Variations"],"data":[]}
                path = BATCH_CONFIG_DIR / profile / f"{dataset}.toml"
                try:
                    cfg=load_config(path)
                except Exception:
                    return {"headers":["Argument","Current","Variations"],"data":[]}
                rows=[[k,str(v),""] for k,v in cfg.items()]
                return {"headers":["Argument","Current","Variations"],"data":rows}

            ds_exp.change(_load_param_table, inputs=[ds_exp, prof_exp], outputs=var_table)
            prof_exp.change(_load_param_table, inputs=[ds_exp, prof_exp], outputs=var_table)

            # ---------- Clear experiments (with confirm via JS) ----------
            def _clear_experiments():
                """Delete all experiment JSON files and return status."""
                rows, sel_upd = _list_experiments()
                from .experiments import EXP_DIR
                import os
                cnt = 0
                for p in EXP_DIR.glob("*.json"):
                    try:
                        os.remove(p)
                        cnt += 1
                    except Exception:
                        pass
                # Refresh again after deletion
                rows, sel_upd = _list_experiments()
                return f"🗑️ Deleted {cnt} experiment(s)", rows, sel_upd

            # Toggle button enable/disable when checkbox changes
            def _toggle_btn(enabled):
                return gr.update(interactive=enabled)

            confirm_chk.change(_toggle_btn, inputs=confirm_chk, outputs=clear_exp_btn)

            clear_exp_btn.click(
                _clear_experiments,
                inputs=[],
                outputs=[exp_msg, exp_table, sel_exp],
            )

            # ---------------- Debug Jobs -----------------
            with gr.Accordion("Debug – Current Jobs", open=False):
                refresh_jobs_btn = gr.Button("Refresh Jobs")
                jobs_df = gr.Dataframe(headers=["ID", "Dataset", "Profile", "Status"], interactive=False, row_count=(5, "dynamic"))

                def _list_jobs_debug():
                    rows = [[j.id, j.dataset, j.profile, j.status] for j in JOB_MANAGER.list_jobs()]
                    return rows

                refresh_jobs_btn.click(lambda: _list_jobs_debug(), inputs=None, outputs=jobs_df)

        # ---------------- Queue Tab (top-level) -----------------
        with gr.Tab("Queue"):
            # Components are defined within the layout below
            # --- Layout: table (left) | stats, log & controls (right) ---
            with gr.Row():
                # LEFT COLUMN – queue overview table
                with gr.Column(scale=6):
                    queue_df = gr.Dataframe(
                        headers=["ID", "Dataset", "Mode", "Status", "Comment"],
                        interactive=False,
                    )

                # RIGHT COLUMN – stats, log and actions
                with gr.Column(scale=4):
                    stats_md = gr.Markdown()
                    refresh_queue_btn = gr.Button("Refresh", variant="secondary")

                    queue_log_box = gr.Textbox(label="Debug log", lines=14, interactive=False)

                    cancel_select = gr.Dropdown(label="Job to cancel (pending/running)")
                    with gr.Row():
                        cancel_job_btn = gr.Button("Cancel job")
                        clear_queue_btn = gr.Button("Clear queue", variant="stop")
                    cancel_msg = gr.Markdown()

            # ---------------- Logic -----------------

            def _refresh_queue():
                jobs = JOB_MANAGER.list_jobs()
                rows = [
                    [
                        j.id,
                        j.dataset,
                        j.profile,
                        j.status,
                        (j.progress_str or ""),
                    ]
                    for j in jobs
                ]
                pendings = [j.id for j in jobs if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]

                # Stats summary
                total = len(jobs)
                running = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
                queued = sum(1 for j in jobs if j.status == JobStatus.PENDING)
                done = sum(1 for j in jobs if j.status == JobStatus.DONE)
                failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
                canceled = sum(1 for j in jobs if j.status == JobStatus.CANCELED)
                stats_text = (
                    f"**Total:** {total} · **Running:** {running} · **Queued:** {queued} · "
                    f"**Done:** {done} · **Failed:** {failed} · **Canceled:** {canceled}"
                )

                debug_lines = [
                    f"{j.id} | {j.dataset} | {j.profile} | {j.status} | {j.progress_str}"
                    for j in jobs
                ]
                debug_text = "\n".join(debug_lines)

                return (
                    gr.update(value=rows),
                    gr.update(choices=pendings),
                    stats_text,
                    gr.update(value=debug_text),
                )

            # Wiring
            refresh_queue_btn.click(
                _refresh_queue,
                inputs=[],
                outputs=[queue_df, cancel_select, stats_md, queue_log_box],
            )

            # Auto-refresh every 2 s
            gr.Timer(2.0).tick(
                _refresh_queue, None, [queue_df, cancel_select, stats_md, queue_log_box]
            )

            def _cancel_job(job_id):
                if not job_id:
                    return "Select a job id", gr.update(), gr.update(), "", gr.update()
                JOB_MANAGER.cancel(job_id)
                queue_upd, select_upd, stats, log_update = _refresh_queue()
                return f"Cancel requested for {job_id}", select_upd, queue_upd, stats, log_update

            cancel_job_btn.click(
                _cancel_job,
                inputs=cancel_select,
                outputs=[cancel_msg, cancel_select, queue_df, stats_md, queue_log_box],
            )

            def _clear_queue():
                """Remove all jobs that are not currently running from the queue."""
                JOB_MANAGER.clear_queue()
                queue_upd, select_upd, stats, log_update = _refresh_queue()
                return "Queue cleared", select_upd, queue_upd, stats, log_update

            clear_queue_btn.click(
                _clear_queue,
                inputs=[],
                outputs=[cancel_msg, cancel_select, queue_df, stats_md, queue_log_box],
            )

        # ---------------- Model Organizer Tab -----------------
        with gr.Tab("Model Organizer"):
            from .run_registry import ensure_initial_scan as _rr_scan, _load as _rr_load  # type: ignore

            _rr_scan()

            # ---------------- Column visibility -----------------
            # Default full set of columns shown in the organizer table
            MO_HEADERS = [
                "JobID",
                "Dataset",
                "Profile",
                "Status",
                "Epochs",
                "Batch",
                "LR",
                "FID",
                "CLIP",
                "Queue",
                "End",
            ]

            # Checkbox selector – user chooses which columns are visible
            mo_cols = gr.CheckboxGroup(
                label="Visible columns",
                choices=MO_HEADERS,
                value=MO_HEADERS,
                interactive=True,
            )

            # ---------------- Search & filter row -----------------
            with gr.Row():
                # Left half – query + status dropdown
                with gr.Column(scale=6):
                    with gr.Row():
                        mo_query = gr.Textbox(label="Search", placeholder="dataset, profile, job_id…")
                        mo_status = gr.Dropdown(label="Status", choices=["all", "pending", "running", "done", "failed", "canceled"], value="all")

                # Right half – action buttons
                with gr.Column(scale=2):
                    mo_refresh = gr.Button("Refresh")
                    mo_clear = gr.Button("🗑️ Clear records", variant="stop")

            def _filter_cols(rows, selected_cols):
                """Return rows keeping only the columns selected by the user."""
                if not selected_cols:
                    return []  # no cols selected → empty table
                idxs = [MO_HEADERS.index(c) for c in selected_cols if c in MO_HEADERS]
                return [[row[i] for i in idxs] for row in rows]

            def _refresh_runs_tbl(query="", status="all"):
                data = _rr_load().values()
                rows = []
                q = ("" if query is None else str(query)).strip().lower()
                for rec in data:
                    if status != "all" and rec.get("status") != status:
                        continue
                    if q and q not in rec.get("dataset", "").lower() and q not in rec.get("profile", "").lower() and q not in rec.get("job_id", ""):
                        continue
                    rows.append([
                        rec.get("job_id"),
                        rec.get("dataset"),
                        rec.get("profile"),
                        rec.get("status"),
                        rec.get("epochs"),
                        rec.get("batch_size"),
                        rec.get("learning_rate"),
                        rec.get("fid"),
                        rec.get("clip"),
                        rec.get("queued_time", "")[:19],
                        rec.get("end_time", "")[:19],
                    ])
                return rows

            mo_table = gr.Dataframe(headers=MO_HEADERS, datatype=["str"]*len(MO_HEADERS), interactive=False, max_height=400)

            # Initial table data
            mo_table.value = _filter_cols(_refresh_runs_tbl(), MO_HEADERS)

            # ----- JobID dropdown (for detail view) -----
            _ids_init = [row[0] for row in _refresh_runs_tbl()]
            sel_job = gr.Dropdown(label="JobID", choices=_ids_init, value=(_ids_init[0] if _ids_init else None))

            def _update_dropdown():
                ids = [row[0] for row in _refresh_runs_tbl(mo_query.value if mo_query else "", mo_status.value if mo_status else "all")]
                return gr.update(choices=ids, value=(ids[0] if ids else None))

            # Sync dropdown initially and after any UI reload
            _sel_upd = _update_dropdown()
            sel_job.choices = [(str(c), c) for c in _sel_upd["choices"]]
            sel_job.value = _sel_upd["value"]

            open_btn = gr.Button("Show details")
            det_md = gr.Markdown()
            det_gallery = gr.Gallery(columns=4, height="auto")

            def _job_detail(job_id):
                if not job_id:
                    return "Select a job", []
                from .run_registry import _load as _rr_load
                recs = _rr_load()
                rec = recs.get(job_id)
                if not rec:
                    return "Job not found", []

                # markdown summary
                md = [f"### {job_id}"]
                for k in ["dataset", "profile", "status", "epochs", "batch_size", "learning_rate", "fid", "clip"]:
                    if k in rec and rec[k] is not None:
                        md.append(f"- **{k}**: {rec[k]}")
                # times
                for t in ["queued_time", "start_time", "end_time"]:
                    if t in rec:
                        md.append(f"- **{t}**: {rec[t]}")
                md_txt = "\n".join(md)

                # gallery: last 8 images
                from pathlib import Path
                run_path = Path(rec["run_dir"])
                sample_dir = run_path / "sample"
                if sample_dir.exists():
                    imgs = sorted(sample_dir.glob("*.png"))[-8:]
                else:
                    # fallback: search recursively for last PNG images
                    imgs = sorted(run_path.rglob("*.png"))[-8:]
                return md_txt, [str(p) for p in imgs]

            # Helper to refresh both table and dropdown based on query & status
            def _refresh_organizer(query_txt, status_sel, cols_sel):
                """Refresh table when query / status / columns change."""
                all_rows = _refresh_runs_tbl(query_txt, status_sel)
                filtered_rows = _filter_cols(all_rows, cols_sel)
                return gr.update(value=filtered_rows, headers=cols_sel, datatype=["str"]*len(cols_sel)), _update_dropdown()

            open_btn.click(_job_detail, inputs=sel_job, outputs=[det_md, det_gallery])

            mo_refresh.click(_refresh_organizer, inputs=[mo_query, mo_status, mo_cols], outputs=[mo_table, sel_job])
            mo_query.submit(_refresh_organizer, inputs=[mo_query, mo_status, mo_cols], outputs=[mo_table, sel_job])
            mo_status.change(_refresh_organizer, inputs=[mo_query, mo_status, mo_cols], outputs=[mo_table, sel_job])
            mo_cols.change(_refresh_organizer, inputs=[mo_query, mo_status, mo_cols], outputs=[mo_table, sel_job])

            # ---------- Clear all registry records ----------
            def _clear_runs():
                from .run_registry import _save as _rr_save  # type: ignore
                _rr_save({})  # overwrite with empty dict
                # Recompute table with current column selection
                rows_all = _refresh_runs_tbl()
                rows_filtered = _filter_cols(rows_all, mo_cols.value or [])
                return gr.update(value=rows_filtered, headers=mo_cols.value or [], datatype=["str"] * len(mo_cols.value or [])), _update_dropdown()

            # Use JS confirm dialog; if user cancels, Python fn is skipped
            mo_clear.click(lambda: None, [], [], js="() => confirm('Delete all records?')").then(
                _clear_runs,
                inputs=None,
                outputs=[mo_table, sel_job],
            )

        # ---------------- Integrations Tab -----------------

        with gr.Tab("Integrations"):
            cfg_init = _load_integrations_cfg()

            with gr.Accordion("Hugging Face Hub", open=False):
                hf_token = gr.Textbox(label="Token", value=cfg_init.get("AUTO_HF_TOKEN", ""), type="password")
                hf_repo = gr.Textbox(label="Repo ID", value=cfg_init.get("AUTO_HF_REPO", ""))
                hf_enable = gr.Checkbox(label="Enable upload", value=cfg_init.get("AUTO_HF_ENABLE", "0") == "1")
                hf_private = gr.Checkbox(label="Private repo", value=cfg_init.get("AUTO_HF_PRIVATE", "0") == "1")

            # --- Amazon S3 integration removed ---

            with gr.Accordion("Remote output", open=False):
                remote_base = gr.Textbox(label="Remote base path", value=_load_integrations_cfg().get("AUTO_REMOTE_BASE", ""))
                remote_direct = gr.Checkbox(label="Write directly to remote path", value=_load_integrations_cfg().get("AUTO_REMOTE_DIRECT", "0") == "1")

            with gr.Accordion("Google Sheets", open=False):
                gr.Markdown("""
                ### 📋 How to set up Google Sheets integration
                
                **Step 1: Create a Google Cloud Project**
                1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                2. Create a new project or select an existing one
                3. Enable the **Google Sheets API** for your project
                
                **Step 2: Create Service Account**
                1. Go to **IAM & Admin > Service Accounts**
                2. Click **"Create Service Account"**
                3. Enter a name (e.g., "AutoTrainV2 Bot")
                4. Click **"Create and Continue"**
                5. Skip the optional steps and click **"Done"**
                
                **Step 3: Generate JSON Key**
                1. Click on your newly created service account
                2. Go to the **"Keys"** tab
                3. Click **"Add Key" > "Create new key"**
                4. Select **JSON** format and click **"Create"**
                5. Save the downloaded JSON file securely (you'll need its path below)
                
                **Step 4: Create Google Sheet**
                1. Create a new [Google Sheet](https://sheets.google.com/)
                2. Copy the **Spreadsheet ID** from the URL:
                   - URL: `https://docs.google.com/spreadsheets/d/`**`1mCl7WTOd1KAp-f73rzGLWZ_ob4KpggBSOzX8CT8WP34`**`/edit`
                   - ID: `1mCl7WTOd1KAp-f73rzGLWZ_ob4KpggBSOzX8CT8WP34`
                
                **Step 5: Share Sheet with Service Account**
                1. Open your Google Sheet
                2. Click **"Share"** button
                3. Add the service account email as an **Editor**
                   - Find the email in your JSON file: `"client_email": "autotrainv2-bot@project.iam.gserviceaccount.com"`
                4. Click **"Send"**
                
                ---
                """)
                
                gsheet_cred = gr.Textbox(
                    label="Service account JSON path", 
                    value=cfg_init.get("AUTO_GSHEET_CRED", ""),
                    placeholder="/path/to/your/service-account-key.json",
                    info="Full path to the JSON key file you downloaded in Step 3"
                )
                gsheet_id = gr.Textbox(
                    label="Spreadsheet ID", 
                    value=cfg_init.get("AUTO_GSHEET_ID", ""),
                    placeholder="1mCl7WTOd1KAp-f73rzGLWZ_ob4KpggBSOzX8CT8WP34",
                    info="The long ID from your Google Sheet URL (see Step 4 above)"
                )
                gsheet_tab = gr.Textbox(
                    label="Worksheet name (optional)", 
                    value=cfg_init.get("AUTO_GSHEET_TAB", ""),
                    placeholder="Sheet1",
                    info="Leave blank to use the first sheet, or specify a tab name"
                )

                # --- Variables checkbox list ---
                from .integrations import GSHEET_HEADER as _GSHEET_HEADER  # local import avoids heavy module at top
                _keys_csv = cfg_init.get("AUTO_GSHEET_KEYS", "") or ""
                _default_keys = [k.strip() for k in _keys_csv.split(",") if k.strip()] if _keys_csv else _GSHEET_HEADER
                gsheet_keys = gr.CheckboxGroup(
                    label="Variables (columns) to include",
                    choices=_GSHEET_HEADER,
                    value=_default_keys,
                    visible=True,
                )

                # Textbox to specify order CSV
                gsheet_order = gr.Textbox(
                    label="Order (comma-separated list)",
                    value=",".join(_default_keys),
                    lines=1,
                )

                def _update_order_txt(selected, current_txt):
                    current_order = [v.strip() for v in (current_txt or "").split(",") if v.strip()]
                    # Keep existing order of still-selected keys
                    new_order = [k for k in current_order if k in selected]
                    # Append newly selected keys
                    for k in selected:
                        if k not in new_order:
                            new_order.append(k)
                    # Remove deselected keys are already excluded
                    return ",".join(new_order)

                gsheet_keys.change(_update_order_txt, inputs=[gsheet_keys, gsheet_order], outputs=gsheet_order)

                # ---- Test Google Sheets (inside accordion) ----
                with gr.Row():
                    test_gs_btn = gr.Button("Test Google Sheets", variant="primary")
                    test_gs_out = gr.Markdown()
                test_gs_debug = gr.Textbox(label="Debug", visible=False, lines=8, interactive=False)

            save_int_btn = gr.Button("Save settings")
            save_int_out = gr.Markdown()

            def _save_integrations(
                hf_token, hf_repo, hf_enable, hf_private,
                remote_base, remote_direct,
                gsheet_cred, gsheet_id, gsheet_tab,
                gsheet_keys, gsheet_order,
            ):
                # Compute final ordered list
                order_list_raw = [v.strip() for v in (gsheet_order or "").split(",") if v.strip()]
                final_keys = [k for k in order_list_raw if k in gsheet_keys]
                for k in gsheet_keys:  # append any missing selected keys
                    if k not in final_keys:
                        final_keys.append(k)

                cfg = {
                    "AUTO_HF_TOKEN": hf_token.strip() or None,
                    "AUTO_HF_REPO": hf_repo.strip() or None,
                    "AUTO_HF_ENABLE": "1" if hf_enable else "0",
                    "AUTO_HF_PRIVATE": "1" if hf_private else "0",
                    "AUTO_REMOTE_BASE": remote_base.strip() or None,
                    "AUTO_REMOTE_DIRECT": "1" if remote_direct else "0",
                    "AUTO_GSHEET_CRED": gsheet_cred.strip() or None,
                    "AUTO_GSHEET_ID": gsheet_id.strip() or None,
                    "AUTO_GSHEET_TAB": gsheet_tab.strip() or None,
                    "AUTO_GSHEET_KEYS": (",".join(final_keys) if final_keys else None),
                }

                _save_integrations_cfg(cfg)
                return "Settings saved ✅"

            save_int_btn.click(
                _save_integrations,
                inputs=[
                    hf_token, hf_repo, hf_enable, hf_private,
                    remote_base, remote_direct,
                    gsheet_cred, gsheet_id, gsheet_tab,
                    gsheet_keys, gsheet_order,
                ],
                outputs=save_int_out,
            )

            # ---------------- Callback de test -----------------

            def _test_gsheet():
                """Envía una fila dummy al Sheet para verificar la conexión."""
                try:
                    from .integrations import _append_to_gsheets  # type: ignore
                    from .job_manager import Job, JobStatus  # type: ignore
                    from pathlib import Path

                    import os
                    cfg = _load_integrations_cfg()
                    cred = cfg.get("AUTO_GSHEET_CRED")
                    sid = cfg.get("AUTO_GSHEET_ID")
                    tab = cfg.get("AUTO_GSHEET_TAB")

                    if not cred or not sid:
                        return "❌ Credenciales o Spreadsheet ID no configurados", ""

                    # Set env vars for current process so integration helper can read them
                    os.environ["AUTO_GSHEET_CRED"] = cred
                    os.environ["AUTO_GSHEET_ID"] = sid
                    if tab:
                        os.environ["AUTO_GSHEET_TAB"] = tab

                    # Job ficticio
                    job = Job(
                        dataset="_test_",
                        profile="Flux",
                        toml_path=Path("/dev/null"),
                        run_dir=Path("/tmp/autotrain_test_run"),
                    )
                    job.status = JobStatus.DONE

                    ok, msg = _append_to_gsheets(job, uploads=["dummy"])
                    dbg = "" if ok else msg  # show full message in debug if failed
                    return ("✅ " if ok else "❌ ") + msg, dbg
                except Exception as e:  # noqa: BLE001
                    import traceback, html
                    tb = traceback.format_exc()
                    return f"❌ Error: {e}", tb

            def _handle_test_click():
                msg, debug = _test_gsheet()
                show_dbg = bool(debug)
                return msg, gr.update(visible=show_dbg, value=debug)

            test_gs_btn.click(_handle_test_click, inputs=[], outputs=[test_gs_out, test_gs_debug])

        # === CONFIGURATION TAB ===
        # (Removed "⚙️ Configuration" tab as per user request — all GPU profile UI code deleted)
        return demo


def launch(**launch_kwargs):  # pragma: no cover
    """Launch the Gradio application.

    Accepts the same kwargs as ``gr.Blocks.launch`` (e.g., share=True).
    """
    # Ensure external dataset folders are allowed so Gradio can serve previews
    from .dataset_sources import load_sources as _ds_load_src

    extra_allowed = [str(p) for p in _ds_load_src()]

    # Add remote output base if configured
    remote_base_env = os.getenv("AUTO_REMOTE_BASE")
    if not remote_base_env:
        try:
            cfg_tmp = _load_integrations_cfg()
            remote_base_env = cfg_tmp.get("AUTO_REMOTE_BASE")
        except Exception:
            pass
    if remote_base_env:
        extra_allowed.append(remote_base_env)

    base_allowed = [str(INPUT_DIR)]
    allow = list(dict.fromkeys(base_allowed + extra_allowed))

    if "allowed_paths" in launch_kwargs and launch_kwargs["allowed_paths"]:
        launch_kwargs["allowed_paths"] = list(dict.fromkeys(launch_kwargs["allowed_paths"] + allow))
    else:
        launch_kwargs["allowed_paths"] = allow

    build_ui().launch(**launch_kwargs)


# -------------------- New helpers --------------------

from .utils.common import (
    list_available_datasets,
    get_dataset_choices_for_ui,
    dataset_file_counts,
    resolve_dataset_path,
    format_dataset_status,
)

def _list_dataset_raw():
    """Return list of dataset names (folders under input)."""
    return list_available_datasets()


def _list_dataset_choices(selected: str | None = None):
    return get_dataset_choices_for_ui(selected)


def _dataset_counts(name: str):
    return dataset_file_counts(name)


def cb_dataset_selected(name: str):
    if not name:
        return gr.update(visible=False), ""
    status = format_dataset_status(name)
    return gr.update(visible=True, value=None), status


def cb_upload_files(dataset: str, files):
    if not dataset:
        return "Select a dataset", ""
    import shutil

    def _extract_path(obj) -> Path | None:
        if obj is None:
            return None
        if isinstance(obj, (str, Path)):
            return Path(obj)
        # Gradio v4 dict
        if isinstance(obj, dict):
            if "path" in obj:
                return Path(obj["path"])
            if "name" in obj and Path(obj["name"]).exists():
                return Path(obj["name"])
        # TemporaryUploadedFile-like
        if hasattr(obj, "path"):
            return Path(obj.path)
        if hasattr(obj, "name") and Path(obj.name).exists():
            return Path(obj.name)
        return None

    dest = INPUT_DIR / dataset
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    allowed_exts = set(SUPPORTED_IMAGE_EXTS) | set(SUPPORTED_TEXT_EXTS)

    for f in files or []:
        src_path = _extract_path(f)
        if src_path is None or not src_path.exists():
            continue
        if src_path.suffix.lower().lstrip(".") not in allowed_exts:
            continue
        target = dest / src_path.name
        try:
            shutil.copy2(src_path, target)
            count += 1
        except Exception:
            pass
    status = format_dataset_status(dataset)
    return f"{count} file(s) uploaded", status


def cb_build_output_single(dataset: str, repeats: int = 30, res_str: str | None = None):
    if not dataset:
        return "Select a dataset"
    import shutil
    # ensure dataset present in input
    src_ext = _ds_find_path(dataset)
    dest = INPUT_DIR / dataset
    if src_ext and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_ext, dest, dirs_exist_ok=True)

    try:
        populate_output_structure_single(dataset, repeats=int(repeats))

        cropped = 0
        if res_str:
            special_folder = OUTPUT_DIR / dataset / "img" / f"{repeats}_{dataset} person"
            cropped = _crop_images_in_folder(special_folder, res_str)

        msg = f"Output structure created/fixed for '{dataset}'."
        if res_str:
            msg += f" Cropped {cropped} image(s) in output folder."
        return msg
    except Exception as e:
        return f"Error: {e}"


# helper for training dropdown

def _list_dataset_in_profile(profile: str):
    subdir = BATCH_CONFIG_DIR / ("Flux" if profile == "Flux" else ("FluxLORA" if profile == "FluxLORA" else "Nude"))
    if not subdir.exists():
        return []
    return [p.stem for p in subdir.glob("*.toml")]


# Cancel current training through JobManager
def cb_cancel_training():
    current_job = JOB_MANAGER.get_current_job()
    if not current_job:
        return "No training running."
    JOB_MANAGER.cancel(current_job.id)
    return f"Cancel requested for job {current_job.id}."


def _cb_gen_presets(dataset):
    if not dataset:
        return "Select a dataset", gr.update()
    try:
        populate_output_structure_single(dataset)
    except Exception:
        pass
    generate_presets_for_dataset(dataset)
    # refresh override dropdown choices
    return f"Presets generated for {dataset}", _list_dataset_choices(dataset)


# ----------------- Queue dashboard helper ------------------

def _format_jobs(jobs):
    rows = []
    for j in jobs:
        rows.append([j.id, j.dataset, j.profile, j.status])
    return rows


# -------------------------------- Integrations helpers ---------------------

from .utils.common import load_integration_config, save_integration_config

def _load_integrations_cfg() -> dict:
    return load_integration_config()

def _save_integrations_cfg(cfg: dict):
    save_integration_config(cfg)


# -------------------------------- GPU Profile helpers ---------------------

GPU_PROFILES_FILE = get_project_root() / "gpu_profiles.json"

def _load_gpu_profiles() -> dict:
    """Load GPU profiles from JSON file."""
    if not GPU_PROFILES_FILE.exists():
        # Create default profiles
        default_profiles = {
            "RTX 4090": {
                "name": "RTX 4090",
                "vram": 24,
                "flux_speed": 0.6,  # seconds per iteration
                "flux_lora_speed": 0.4,
                "nude_speed": 0.5,
                "power_consumption": 450,
                "is_active": True
            },
            "RTX 4080": {
                "name": "RTX 4080",
                "vram": 16,
                "flux_speed": 0.8,
                "flux_lora_speed": 0.5,
                "nude_speed": 0.6,
                "power_consumption": 320,
                "is_active": False
            },
            "RTX 3090": {
                "name": "RTX 3090",
                "vram": 24,
                "flux_speed": 0.9,
                "flux_lora_speed": 0.6,
                "nude_speed": 0.7,
                "power_consumption": 350,
                "is_active": False
            }
        }
        _save_gpu_profiles(default_profiles)
        return default_profiles
    
    try:
        with open(GPU_PROFILES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_gpu_profiles(profiles: dict):
    """Save GPU profiles to JSON file."""
    try:
        with open(GPU_PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving GPU profiles: {e}")

def _get_active_gpu_profile():
    """Get the currently active GPU profile."""
    profiles = _load_gpu_profiles()
    for profile_id, profile in profiles.items():
        if profile.get("is_active", False):
            return profile
    # If no active profile, return the first one or default
    if profiles:
        return list(profiles.values())[0]
    return {
        "name": "Default GPU",
        "vram": 16,
        "flux_speed": 0.8,
        "flux_lora_speed": 0.5,
        "nude_speed": 0.6,
        "power_consumption": 300,
        "is_active": True
    }

def _set_active_gpu_profile(profile_name: str):
    """Set a GPU profile as active."""
    profiles = _load_gpu_profiles()
    # Deactivate all profiles
    for profile in profiles.values():
        profile["is_active"] = False
    # Activate the selected profile
    if profile_name in profiles:
        profiles[profile_name]["is_active"] = True
        _save_gpu_profiles(profiles)
        return True
    return False


# Resolve dataset folder (input or external)
def _resolve_dataset_path(name: str) -> Path | None:
    return resolve_dataset_path(name)


if __name__ == "__main__":  # pragma: no cover
    launch() 
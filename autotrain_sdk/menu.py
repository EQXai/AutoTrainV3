from __future__ import annotations
"""Interactive CLI menu based on *questionary* (Phase 1).

It allows a non-technical user to perform the most common tasks that are
available in the Gradio interface directly from a terminal – navigating with
arrow keys and selecting options.

Quick usage::

    python -m autotrain_sdk.menu

(In production an *entry-point* `autotrain menu` will be provided.)
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import sys
import os
import shutil
import toml

import questionary as q
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
import time
from rich.text import Text
from rich.console import Group
from rich.padding import Padding

from .dataset import (
    create_input_folders,
    populate_output_structure,
    clean_workspace,
    populate_output_structure_single,
)
from .configurator import (
    generate_presets,
    generate_presets_for_dataset,
    load_config,
    save_config,
    update_config,
)
from .trainer import run_training
from .gradio_app import launch as launch_web, _load_integrations_cfg as _load_int_cfg, _save_integrations_cfg as _save_int_cfg
from autotrain_sdk.paths import LOGS_DIR
from .job_manager import JobManager, JobStatus, Job

console = Console()

# Supported image extensions
SUPPORTED_IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp'}

# ------------------------------
# Integration toggle helpers (shared with CLI)
# ------------------------------

INTEGRATION_FLAGS = {
    "gsheet": "AUTO_GSHEET_ENABLE",
    "hf": "AUTO_HF_ENABLE",
    "remote": "AUTO_REMOTE_ENABLE",
}

def _integration_status(name: str) -> bool:
    """Return True if integration *name* is enabled (env var or stored cfg)."""
    var = INTEGRATION_FLAGS.get(name.lower())
    if not var:
        return False
    cfg = _load_int_cfg()
    return (os.getenv(var) or str(cfg.get(var, "0"))) == "1"

# ---------------------------------------------------------------------------
# Progress bar helpers
# ---------------------------------------------------------------------------

def _create_progress_bar(title: str = "Processing"):
    """Creates a custom progress bar with enhanced styling"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    )

def _progress_operation(items, operation_func, title="Processing", item_name="item"):
    """
    Runs an operation on multiple items with a progress bar
    
    Args:
        items: List of items to process
        operation_func: Function that takes an item and returns (success, message)
        title: Title of the operation
        item_name: Name of the item type (to display in the bar)
    
    Returns:
        List of results from operation_func
    """
    results = []
    
    with _create_progress_bar(title) as progress:
        task = progress.add_task(f"[green]{title}...", total=len(items))
        
        for i, item in enumerate(items, 1):
            # Update description with the current item
            progress.update(task, description=f"[green]{title} {item_name} {i}/{len(items)}")
            
            try:
                result = operation_func(item)
                results.append(result)
                
                # Simulate processing time to show the bar
                time.sleep(0.1)
                
            except Exception as e:
                results.append((False, f"Error: {str(e)}"))
            
            progress.advance(task)
    
    return results

def _progress_with_steps(steps, title="Processing"):
    """
    Runs a series of steps with a progress bar
    
    Args:
        steps: List of tuples (step_name, step_func)
        title: Title of the operation
    
    Returns:
        List of results from step functions
    """
    results = []
    
    with _create_progress_bar(title) as progress:
        task = progress.add_task(f"[green]{title}...", total=len(steps))
        
        for step_name, step_func in steps:
            progress.update(task, description=f"[green]{step_name}")
            
            try:
                result = step_func()
                results.append((True, result))
                
                # Simulate processing time
                time.sleep(0.2)
                
            except Exception as e:
                results.append((False, f"Error in {step_name}: {str(e)}"))
            
            progress.advance(task)
    
    return results

# ---------------------------------------------------------------------------
# Enhanced UI helpers (navigation, headers, context info)
# ---------------------------------------------------------------------------

def _get_system_stats():
    """Gets system statistics to display in headers"""
    from autotrain_sdk.paths import BATCH_CONFIG_DIR
    
    # Use cached dataset info to avoid duplication
    try:
        datasets_info = _get_datasets_cached()
        input_datasets = len(datasets_info['input'])
        external_datasets = len(datasets_info['external'])
        # Count unique datasets (avoids counting the same dataset twice)
        total_datasets = len(datasets_info['all'])
        
        # Calculate pending datasets (external datasets not yet in input/)
        input_set = set(datasets_info['input'])
        external_set = set(datasets_info['external'])
        pending_datasets = len(external_set - input_set)
        
    except Exception:
        # Fallback to direct counting if cache fails
        from autotrain_sdk.paths import INPUT_DIR
        from autotrain_sdk.dataset_sources import list_external_datasets
        
        input_dataset_names = [p.name for p in INPUT_DIR.glob("*") if p.is_dir()]
        external_dataset_names = list(list_external_datasets().keys())
        
        input_datasets = len(input_dataset_names)
        external_datasets = len(external_dataset_names)
        # Use set to eliminate duplicates
        total_datasets = len(set(input_dataset_names + external_dataset_names))
        
        # Calculate pending datasets
        input_set = set(input_dataset_names)
        external_set = set(external_dataset_names)
        pending_datasets = len(external_set - input_set)
    
    # Count presets
    presets = len(list(BATCH_CONFIG_DIR.glob("*/*.toml")))
    
    # Detailed job information
    jobs = _JOB_MANAGER.list_jobs()
    running_jobs = [j for j in jobs if j.status == JobStatus.RUNNING]
    pending_jobs = [j for j in jobs if j.status == JobStatus.PENDING]
    
    # Get ETA of the running job
    current_job_eta = None
    current_job_progress = None
    if running_jobs:
        current_job = running_jobs[0]  # First running job
        current_job_eta = getattr(current_job, 'eta', None)
        current_job_progress = getattr(current_job, 'progress_str', None)
    
    return {
        "input_datasets": input_datasets,
        "external_datasets": external_datasets,
        "total_datasets": total_datasets,
        "pending_datasets": pending_datasets,
        "presets": presets,
        "active_jobs": len(running_jobs) + len(pending_jobs),
        "running_jobs": len(running_jobs),
        "pending_jobs": len(pending_jobs),
        "current_job_eta": current_job_eta,
        "current_job_progress": current_job_progress
    }

def _header_with_context(title: str, breadcrumbs: list[str] = None, show_stats: bool = True):
    """Clean and tidy header - Phase 4"""
    console.clear()
    
    # Simplified breadcrumbs
    if breadcrumbs and len(breadcrumbs) > 1:
        path = " > ".join(breadcrumbs)
        console.print(f"[dim cyan]{path}[/dim cyan]")
        console.print()
    
    # Main title with better styling
    console.print(f"[bold white on blue] {title} [/bold white on blue]")
    console.print()
    
    # Simplified and clear statistics
    if show_stats:
        try:
            stats = _get_system_stats()
            # Show essential info clearly with separation between available and pending datasets
            info_parts = []
            if stats['input_datasets'] > 0:
                info_parts.append(f"Datasets: {stats['input_datasets']}")
            if stats['pending_datasets'] > 0:
                info_parts.append(f"Pending datasets: {stats['pending_datasets']}")
            if stats['presets'] > 0:
                info_parts.append(f"Presets: {stats['presets']}")
            
            # Detailed job information
            if stats['running_jobs'] > 0:
                if stats['pending_jobs'] > 0:
                    info_parts.append(f"Jobs: {stats['running_jobs']} running, {stats['pending_jobs']} queued")
                else:
                    info_parts.append(f"Jobs: {stats['running_jobs']} running")
            elif stats['pending_jobs'] > 0:
                info_parts.append(f"Jobs: {stats['pending_jobs']} queued")
            
            if info_parts:
                # Create box content
                info_content = ' • '.join(info_parts)
                
                # Add current job info if it exists
                job_info = ""
                if stats['current_job_eta'] or stats['current_job_progress']:
                    eta_info = []
                    if stats['current_job_progress']:
                        # If progress_str already contains ETA, don't add ETA separately
                        if stats['current_job_eta'] and stats['current_job_eta'] in stats['current_job_progress']:
                            eta_info.append(f"Progress: {stats['current_job_progress']}")
                        else:
                            eta_info.append(f"Progress: {stats['current_job_progress']}")
                            if stats['current_job_eta']:
                                eta_info.append(f"ETA: {stats['current_job_eta']}")
                    elif stats['current_job_eta']:
                        eta_info.append(f"ETA: {stats['current_job_eta']}")
                    
                    if eta_info:
                        job_info = f"\nCurrent job: {' • '.join(eta_info)}"
                
                # Create Panel with the info using the same style as the menu
                full_content = info_content + job_info
                info_panel = Panel(
                    full_content,
                    style="dim",
                    border_style="dim blue",
                    padding=(0, 1)
                )
                console.print(info_panel)
                console.print()
        except Exception:
            pass

def _show_datasets_overview():
    """Muestra resumen detallado de datasets locales y externos"""
    try:
        datasets_info = _get_datasets_cached()
        input_datasets = datasets_info['input']
        external_datasets = datasets_info['external']
        
        # Mostrar datasets disponibles (en input/)
        if input_datasets:
            console.print(f"📁 Available datasets: {len(input_datasets)}")
            # Mostrar ejemplos de datasets locales
            examples = ", ".join(input_datasets[:5])
            if len(input_datasets) > 5:
                examples += f" (+{len(input_datasets)-5} more)"
            console.print(f"[dim]   Ready to use: {examples}[/dim]")
        else:
            console.print("📁 Available datasets: 0")
            console.print("[dim]   No datasets in input/ folder[/dim]")
        
        # Calcular y mostrar datasets pendientes
        if external_datasets:
            external_not_transferred = [ext for ext in external_datasets if ext not in input_datasets]
            
            if external_not_transferred:
                console.print(f"⏳ Pending datasets: {len(external_not_transferred)}")
                # Mostrar ejemplos de externos no transferidos
                ext_examples = ", ".join(external_not_transferred[:3])
                if len(external_not_transferred) > 3:
                    ext_examples += f" (+{len(external_not_transferred)-3} more)"
                console.print(f"[dim]   Need transfer: {ext_examples}[/dim]")
            else:
                console.print("⏳ Pending datasets: 0")
                console.print("[dim]   All external datasets transferred[/dim]")
        else:
            console.print("⏳ Pending datasets: 0")
            console.print("[dim]   No external sources configured[/dim]")
        
        console.print()
    except Exception:
        pass

def _show_presets_overview():
    """Muestra resumen simple de presets disponibles"""
    from autotrain_sdk.paths import BATCH_CONFIG_DIR
    
    profiles = ["Flux", "FluxLORA", "Nude"]
    profile_counts = {}
    
    # Contar presets por perfil
    for profile in profiles:
        presets = list((BATCH_CONFIG_DIR / profile).glob("*.toml"))
        profile_counts[profile] = len(presets)
    
    total_presets = sum(profile_counts.values())
    if total_presets > 0:
        parts = []
        for profile, count in profile_counts.items():
            if count > 0:
                parts.append(f"{profile}: {count}")
        
        if parts:
            console.print(f"Available presets: {', '.join(parts)}")
            console.print()
    else:
        console.print("[dim]No presets configured yet[/dim]")
        console.print()

def _show_jobs_overview():
    """Muestra resumen simple de jobs en cola"""
    jobs = _JOB_MANAGER.list_jobs()
    
    if not jobs:
        return
    
    # Contar por estado
    status_counts = {}
    for job in jobs:
        status = job.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Solo mostrar estados importantes de forma simple
    important_status = []
    if status_counts.get("running", 0) > 0:
        important_status.append(f"Running: {status_counts['running']}")
    if status_counts.get("pending", 0) > 0:
        important_status.append(f"Pending: {status_counts['pending']}")
    if status_counts.get("failed", 0) > 0:
        important_status.append(f"Failed: {status_counts['failed']}")
    
    if important_status:
        console.print(f"Jobs: {', '.join(important_status)}")
        console.print()

# ---------------------------------------------------------------------------
# Help system and shortcuts
# ---------------------------------------------------------------------------

def _show_help():
    """Muestra ayuda rápida del sistema"""
    help_text = """
    [bold]⌨️  Quick Navigation:[/bold]
    • Use number keys (1-9) for quick selection
    • Arrow keys ↑↓ to navigate menus
    • Enter to select, Escape to cancel
    • Type 'h' for help, 'q' to quit/back
    
    [bold]📊 Status Indicators:[/bold]
    • ✓ Available/Complete     • ✗ Missing/Incomplete  
    • 🔄 In Progress          • ⚠️  Warning/Attention
    • ❌ Error/Failed         • 📋 Pending/Queue
    
    [bold]🎯 Common Workflows:[/bold]
    • New dataset: 1→1 (create) → 1→2 (build) → 2→1 (generate presets)
    • Quick train: 4 (QuickTrain 1-Step) → select dataset → configure
    • Monitor: 5 (Jobs) → 4 (live view) to watch progress
    • Check logs: 6 (Logs) → select training log → view/follow
    
    [bold]🔧 Tips:[/bold]
    • External datasets are auto-detected from sources
    • Presets are generated based on dataset structure
    • Jobs run in background - check logs for details
    """
    console.print(Panel(help_text, title="AutoTrainV2 Help", style="blue", padding=(1, 2)))
    _pause()

def _show_gpu_info():
    """Muestra información de GPUs disponibles"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if not gpus:
            console.print("[yellow]No GPUs detected[/yellow]")
            return
        
        table = Table("ID", "Name", "Memory", "Load", "Temperature", box=None)
        for gpu in gpus:
            memory = f"{gpu.memoryFree:.0f}MB / {gpu.memoryTotal:.0f}MB"
            load = f"{gpu.load*100:.1f}%"
            temp = f"{gpu.temperature}°C" if gpu.temperature else "N/A"
            table.add_row(str(gpu.id), gpu.name, memory, load, temp)
        
        console.print(table)
    except ImportError:
        console.print("[dim]GPUtil not available - install with: pip install GPUtil[/dim]")
    except Exception as e:
        console.print(f"[yellow]GPU info error: {e}[/yellow]")

def _show_system_status():
    """Muestra estado detallado del sistema con información de debug"""
    from autotrain_sdk.paths import INPUT_DIR, OUTPUT_DIR, BATCH_CONFIG_DIR, LOGS_DIR
    import os
    import sys
    import platform
    import subprocess
    
    # ========================================
    # 1. VERSIONES Y DEPENDENCIAS
    # ========================================
    console.print("[bold blue]🔍 System Debug Information[/bold blue]")
    console.print()
    
    # Versión de Python
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"[bold]🐍 Python:[/bold] {python_version} ({platform.python_implementation()})")
    
    # Versión de PyTorch
    try:
        import torch
        torch_version = torch.__version__
        console.print(f"[bold]🔥 PyTorch:[/bold] {torch_version}")
        
        # Información adicional de PyTorch
        torch_cuda_available = torch.cuda.is_available()
        torch_cuda_version = torch.version.cuda if torch_cuda_available else "N/A"
        console.print(f"[bold]⚡ PyTorch CUDA:[/bold] {torch_cuda_version} (Available: {'✓' if torch_cuda_available else '✗'})")
    except ImportError:
        console.print("[bold]🔥 PyTorch:[/bold] [red]NOT INSTALLED[/red]")
    
    # Versión de CUDA del sistema
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract CUDA version from nvcc output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    cuda_version = line.split('release')[1].split(',')[0].strip()
                    console.print(f"[bold]🎯 CUDA (System):[/bold] {cuda_version}")
                    break
        else:
            console.print("[bold]🎯 CUDA (System):[/bold] [yellow]nvcc not found[/yellow]")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("[bold]🎯 CUDA (System):[/bold] [yellow]nvcc not found[/yellow]")
    
    console.print()
    
    # ========================================
    # 2. DEPENDENCIAS DEL REQUIREMENTS.TXT
    # ========================================
    console.print("[bold blue]📦 Dependencies Check[/bold blue]")
    console.print()
    
    # Verificar requirements.txt
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        try:
            with open(requirements_path, 'r') as f:
                requirements = f.read().strip().split('\n')
            
            # Filtrar líneas vacías y comentarios
            requirements = [line.strip() for line in requirements if line.strip() and not line.strip().startswith('#')]
            
            console.print(f"[bold]📋 Requirements file:[/bold] Found ({len(requirements)} packages)")
            
            # Verificar cada dependencia
            missing_packages = []
            installed_packages = []
            
            for req in requirements:
                # Extraer nombre del paquete (antes de ==, >=, etc.)
                package_name = req.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('!=')[0].strip()
                
                try:
                    __import__(package_name.replace('-', '_'))
                    installed_packages.append(package_name)
                except ImportError:
                    # Algunos paquetes tienen nombres diferentes para import
                    special_cases = {
                        'pillow': 'PIL',
                        'opencv-python': 'cv2',
                        'scikit-learn': 'sklearn',
                        'beautifulsoup4': 'bs4',
                        'pyyaml': 'yaml',
                        'torch': 'torch',
                        'torchvision': 'torchvision',
                        'torchaudio': 'torchaudio'
                    }
                    
                    import_name = special_cases.get(package_name.lower(), package_name)
                    try:
                        __import__(import_name)
                        installed_packages.append(package_name)
                    except ImportError:
                        missing_packages.append(package_name)
            
            # Mostrar resultados
            if missing_packages:
                console.print(f"[bold]❌ Missing packages ({len(missing_packages)}):[/bold] [red]{', '.join(missing_packages)}[/red]")
            else:
                console.print(f"[bold]✅ All packages installed:[/bold] [green]{len(installed_packages)}/{len(requirements)} packages[/green]")
            
            if missing_packages:
                console.print(f"[dim]Run: pip install {' '.join(missing_packages)}[/dim]")
            
        except Exception as e:
            console.print(f"[bold]📋 Requirements check:[/bold] [red]Error reading file: {e}[/red]")
    else:
        console.print("[bold]📋 Requirements file:[/bold] [yellow]requirements.txt not found[/yellow]")
    
    console.print()
    
    # ========================================
    # 3. RECURSOS DEL SISTEMA
    # ========================================
    console.print("[bold blue]💾 System Resources[/bold blue]")
    console.print()
    
    # Espacio en disco
    def get_disk_usage(path):
        try:
            stat = os.statvfs(path)
            free = stat.f_bavail * stat.f_frsize
            total = stat.f_blocks * stat.f_frsize
            return free, total
        except:
            return 0, 0
    
    free_space, total_space = get_disk_usage(".")
    
    # Estadísticas de archivos
    input_files = len(list(INPUT_DIR.glob("**/*"))) if INPUT_DIR.exists() else 0
    output_files = len(list(OUTPUT_DIR.glob("**/*"))) if OUTPUT_DIR.exists() else 0
    log_files = len(list(LOGS_DIR.glob("*.log"))) if LOGS_DIR.exists() else 0
    
    resource_table = Table("Resource", "Status", "Details", box=None)
    resource_table.add_row("💾 Disk Space", f"{free_space/(1024**3):.1f}GB free", f"of {total_space/(1024**3):.1f}GB total")
    resource_table.add_row("📁 Input Files", str(input_files), "images, captions, configs")
    resource_table.add_row("📤 Output Files", str(output_files), "processed datasets, models")
    resource_table.add_row("📄 Log Files", str(log_files), "training logs, errors")
    
    console.print(resource_table)
    console.print()
    
    # ========================================
    # 4. INFORMACIÓN DE GPU
    # ========================================
    console.print("[bold blue]🎮 GPU Information[/bold blue]")
    console.print()
    _show_gpu_info()

# ---------------------------------------------------------------------------
# Advanced Phase 3 features (User config, search, shortcuts, caching)
# ---------------------------------------------------------------------------

import json
from pathlib import Path
import time
from functools import lru_cache

# User preferences system
USER_CONFIG_FILE = Path.home() / ".autotrain_config.json"

def _load_user_config():
    """Carga configuración del usuario"""
    default_config = {
        "theme": "default",
        "shortcuts_enabled": True,
        "auto_refresh": True,
        "default_profile": "FluxLORA",
        "recent_datasets": [],
        "favorite_actions": [],
        "show_tips": True,
        "quick_mode": False
    }
    
    try:
        if USER_CONFIG_FILE.exists():
            with open(USER_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults for new keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
    except Exception:
        pass
    
    return default_config

def _save_user_config(config):
    """Guarda configuración del usuario"""
    try:
        with open(USER_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass

def _add_recent_dataset(dataset_name):
    """Agrega dataset a la lista de recientes"""
    config = _load_user_config()
    recent = config.get("recent_datasets", [])
    
    # Remove if exists and add to front
    if dataset_name in recent:
        recent.remove(dataset_name)
    recent.insert(0, dataset_name)
    
    # Keep only last 10
    config["recent_datasets"] = recent[:10]
    _save_user_config(config)

def _smart_search(items, prompt="Search"):
    """Búsqueda incremental para listas largas"""
    if len(items) <= 5:
        return q.select(prompt, choices=items).ask()
    
    # For large lists, enable search
    search_term = q.text(f"{prompt} (type to filter, Enter for all):").ask()
    if not search_term:
        return q.select(prompt, choices=items).ask()
    
    # Filter items
    filtered = [item for item in items if search_term.lower() in item.lower()]
    
    if not filtered:
        console.print(f"[yellow]No items match '{search_term}'[/yellow]")
        return None
    
    if len(filtered) == 1:
        return filtered[0]
    
    return q.select(f"Filtered results for '{search_term}'", choices=filtered).ask()

def _quick_action_menu():
    """Menú de acciones rápidas basado en favoritos y recientes"""
    config = _load_user_config()
    recent_datasets = config.get("recent_datasets", [])
    
    if not recent_datasets:
        console.print("[yellow]No recent datasets found[/yellow]")
        
        # Ofrecer inicializar con datasets existentes
        try:
            datasets_info = _get_datasets_cached()
            available = datasets_info['all'][:10]  # Limitar a 10 para no abrumar
            
            if available:
                console.print(f"[dim]Found {len(available)} available datasets. Would you like to add some as favorites?[/dim]")
                if q.confirm("Add some datasets to Quick Actions?", default=True).ask():
                    selected = q.checkbox(
                        "Select datasets to add as favorites:",
                        choices=[q.Choice(ds, checked=True) for ds in available[:5]]  # Pre-select first 5
                    ).ask()
                    
                    if selected:
                        # Agregar datasets seleccionados como recientes
                        for ds in reversed(selected):  # Reverse para mantener el orden
                            _add_recent_dataset(ds)
                        
                        console.print(f"[green]✅ Added {len(selected)} datasets to Quick Actions![/green]")
                        _pause()
                        
                        # Llamar recursivamente para mostrar el menú con datasets
                        return _quick_action_menu()
        except Exception:
            pass
        
        _pause()
        return
    
    actions = [
        f"🚀 QuickTrain: {ds}" for ds in recent_datasets[:3]
    ] + [
        f"🔧 Manage: {ds}" for ds in recent_datasets[:3]
    ] + [
        "📊 View system status",
        "🔄 Refresh caches",
        "⚙️ Configure preferences"
    ]
    
    action = q.select("Quick Actions (recent datasets + shortcuts):", choices=actions + ["Back"]).ask()
    
    if action == "Back":
        return
    elif action.startswith("🚀 QuickTrain:"):
        dataset = action.split(": ")[1]
        _quicktrain_for_dataset(dataset)
    elif action.startswith("🔧 Manage:"):
        dataset = action.split(": ")[1]
        _manage_dataset_direct(dataset)
    elif action == "📊 View system status":
        _show_system_status()
        _pause()
    elif action == "🔄 Refresh caches":
        _clear_caches()
        console.print("[green]Caches cleared[/green]")
        _pause()
    elif action == "⚙️ Configure preferences":
        _user_preferences_menu()

def _quicktrain_for_dataset(dataset_name):
    """QuickTrain optimizado para dataset específico"""
    _add_recent_dataset(dataset_name)
    console.print(f"[cyan]🚀 Starting QuickTrain for {dataset_name}...[/cyan]")
    
    # Load user preferences
    config = _load_user_config()
    default_profile = config.get("default_profile", "FluxLORA")
    
    # Quick mode uses defaults without prompting
    if config.get("quick_mode", False):
        console.print(f"[dim]Using quick mode with {default_profile} profile[/dim]")
        # Here would be the actual QuickTrain logic with defaults
        console.print(f"[green]✅ QuickTrain queued for {dataset_name} with {default_profile}[/green]")
    else:
        console.print(f"[yellow]Redirecting to full QuickTrain interface...[/yellow]")
        # Redirect to full QuickTrain with pre-selected dataset
    
    _pause()

def _manage_dataset_direct(dataset_name):
    """Manejo directo de dataset específico"""
    _add_recent_dataset(dataset_name)
    console.print(f"[cyan]🔧 Managing dataset: {dataset_name}[/cyan]")
    
    # Quick access to most common management tasks
    _manage_existing_dataset_impl(dataset_name)

# Caching system for performance
_cache = {}
_cache_timestamps = {}
CACHE_DURATION = 30  # seconds

@lru_cache(maxsize=128)
def _get_datasets_cached():
    """Cache de datasets disponibles"""
    from autotrain_sdk.paths import INPUT_DIR
    from autotrain_sdk.dataset_sources import list_external_datasets
    
    input_datasets = [p.name for p in INPUT_DIR.glob("*") if p.is_dir()]
    external_datasets = list(list_external_datasets().keys())
    
    return {
        "input": input_datasets,
        "external": external_datasets,
        "all": sorted(set(input_datasets + external_datasets))
    }

def _clear_caches():
    """Limpia todas las caches"""
    global _cache, _cache_timestamps
    _cache.clear()
    _cache_timestamps.clear()
    _get_datasets_cached.cache_clear()

def _user_preferences_menu():
    """Menú de configuración de preferencias del usuario"""
    config = _load_user_config()
    
    while True:
        _header_with_context("User Preferences", ["AutoTrainV2", "Preferences"])
        
        # Show current settings
        table = Table("Setting", "Current Value", "Description", box=None)
        table.add_row("Default Profile", config["default_profile"], "Default training profile")
        table.add_row("Quick Mode", "✓" if config["quick_mode"] else "✗", "Skip confirmations")
        table.add_row("Show Tips", "✓" if config["show_tips"] else "✗", "Show helpful tips")
        table.add_row("Auto Refresh", "✓" if config["auto_refresh"] else "✗", "Auto-refresh data")
        table.add_row("Shortcuts", "✓" if config["shortcuts_enabled"] else "✗", "Enable keyboard shortcuts")
        
        console.print(table)
        console.print()
        
        action = q.select(
            "Choose setting to modify:",
            choices=[
                q.Choice("1. Default Profile", "profile"),
                q.Choice("2. Quick Mode", "quick"),
                q.Choice("3. Show Tips", "tips"),
                q.Choice("4. Auto Refresh", "refresh"),
                q.Choice("5. Shortcuts", "shortcuts"),
                q.Separator(),
                q.Choice("6. Reset to defaults", "reset"),
                q.Choice("7. Export config", "export"),
                q.Choice("8. Import config", "import"),
                q.Separator(),
                q.Choice("0. Back to main menu", "back"),
            ],
        ).ask()
        
        if action == "back":
            break
        elif action == "profile":
            new_profile = q.select("Default training profile:", 
                                 choices=["Flux", "FluxLORA", "Nude"], 
                                 default=config["default_profile"]).ask()
            if new_profile:
                config["default_profile"] = new_profile
        elif action == "quick":
            config["quick_mode"] = q.confirm("Enable quick mode (skip confirmations)?", 
                                           default=config["quick_mode"]).ask()
        elif action == "tips":
            config["show_tips"] = q.confirm("Show helpful tips?", 
                                          default=config["show_tips"]).ask()
        elif action == "refresh":
            config["auto_refresh"] = q.confirm("Auto-refresh data in menus?", 
                                             default=config["auto_refresh"]).ask()
        elif action == "shortcuts":
            config["shortcuts_enabled"] = q.confirm("Enable keyboard shortcuts?", 
                                                   default=config["shortcuts_enabled"]).ask()
        elif action == "reset":
            if q.confirm("Reset all preferences to defaults?", default=False).ask():
                config = _load_user_config()  # This loads defaults
                console.print("[green]Preferences reset to defaults[/green]")
                _pause()
        elif action == "export":
            export_path = q.path("Export config to file:").ask()
            if export_path:
                try:
                    with open(export_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    console.print(f"[green]Config exported to {export_path}[/green]")
                except Exception as e:
                    console.print(f"[red]Export failed: {e}[/red]")
                _pause()
        elif action == "import":
            import_path = q.path("Import config from file:").ask()
            if import_path and Path(import_path).exists():
                try:
                    with open(import_path, 'r') as f:
                        imported_config = json.load(f)
                    config.update(imported_config)
                    console.print(f"[green]Config imported from {import_path}[/green]")
                except Exception as e:
                    console.print(f"[red]Import failed: {e}[/red]")
                _pause()
        
        _save_user_config(config)

def _smart_confirm(message, default=False, quick_mode_key="quick_mode"):
    """Confirmación inteligente que respeta quick_mode"""
    config = _load_user_config()
    if config.get(quick_mode_key, False):
        return default
    return q.confirm(message, default=default).ask()

def _show_tip(tip_text, context="general"):
    """Muestra tips de forma selectiva y menos intrusiva"""
    config = _load_user_config()
    if not config.get("show_tips", True):
        return
    
    # Solo mostrar tips ocasionalmente para no abrumar
    import random
    if random.random() > 0.3:  # Solo 30% de las veces
        return
    
    console.print(f"[dim]Tip: {tip_text}[/dim]")
    console.print()

# ---------------------------------------------------------------------------
# UI helpers (header & pause)
# ---------------------------------------------------------------------------

def _header(title: str):
    """Clear screen and show a boxed header at top."""
    console.clear()
    console.print(Panel(title, style="bold cyan", padding=(0, 2)))

def _pause() -> None:
    """Wait until the user presses Enter."""

    input("\nPress Enter to continue…")


def _error(msg: str):
    console.print(f"[red]ERROR:[/red] {msg}")
    _pause()

# ---------------------------------------------------------------------------
# Dataset flows
# ---------------------------------------------------------------------------

def _datasets_menu():
    while True:
        _header_with_context("Datasets", ["AutoTrainV2", "Datasets"])
        
        # Mostrar información contextual simplificada
        _show_datasets_overview()
        
        # Mostrar datasets recientes si existen
        recent_config = _load_user_config()
        recent = recent_config.get("recent_datasets", [])
        if recent:
            console.print(f"[dim]Recent: {', '.join(recent[:3])}[/dim]")
            console.print()
        
        choice = q.select(
            "Choose action:",
            choices=[
                # 📥 GET DATASETS
                q.Separator("╔═══════════════════╗"),
                q.Separator("║  📥 GET DATASETS  ║"),
                q.Separator("╚═══════════════════╝"),
                q.Choice(" » 1. Import from external sources", "import"),
                q.Choice("   2. Create empty folders", "create"),
                q.Choice("   3. Configure external sources", "sources"),
                q.Separator("─────────────────────"),
                
                # 🔧 PREPARE DATASETS  
                q.Separator("╔═══════════════════════╗"),
                q.Separator("║  🔧 PREPARE DATASETS  ║"),
                q.Separator("╚═══════════════════════╝"),
                q.Choice("   4. Manage individual dataset", "manage"),
                q.Choice("   5. Bulk prepare multiple datasets", "bulk"),
                q.Choice("   6. Build output structure (all)", "build"),
                q.Separator("─────────────────────────"),
                
                # 🗑️ CLEANUP
                q.Separator("╔═══════════════╗"),
                q.Separator("║  🗑️ CLEANUP  ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   7. Clean workspace", "clean"),
                q.Separator("─────────────────"),
                
                q.Choice("   0. Back to main menu", "back"),
            ],
        ).ask()

        if choice == "create":
            names = q.text("Names (separated by comma):").ask()
            if not names:
                continue
            created = create_input_folders([n.strip() for n in names.split(",") if n.strip()])
            console.print(f"[green]Created {len(created)} folders[/green]")
            _pause()

        elif choice == "build":
            min_imgs = q.text("Minimum number of images (0 = no check):", default="0").ask()
            # Ask user for number of repetitions when building the output structure
            repeats = q.text("Repetitions (times each image repeats):", default="30").ask()
            try:
                populate_output_structure(min_images=int(min_imgs), repeats=int(repeats))
                console.print("[green]Structure created successfully[/green]")
            except Exception as e:
                _error(str(e))
            _pause()

        elif choice == "clean":
            # Smart confirmation for destructive operations
            if not _smart_confirm("⚠️  This will delete folders permanently. Continue?", default=False):
                continue
                
            opts = q.checkbox(
                "Select which folders to delete:",
                choices=[
                    q.Choice("input/", checked=True),
                    q.Choice("output/", checked=True),
                    q.Choice("BatchConfig/", checked=True),
                ],
            ).ask()
            
            if opts and _smart_confirm(f"Really delete {', '.join(opts)}?", default=False):
                clean_workspace(
                    delete_input="input/" in opts,
                    delete_output="output/" in opts,
                    delete_batchconfig="BatchConfig/" in opts,
                )
                console.print("[green]Workspace cleaned[/green]")
            else:
                console.print("[yellow]Operation cancelled[/yellow]")
            _pause()

        elif choice == "manage":
            _manage_existing_dataset_smart()

        elif choice == "sources":
            _sources_menu()

        elif choice == "import":
            _import_external_dataset()

        elif choice == "bulk":
            _bulk_prepare_menu()

        else:  # Back
            break

# ---------------------------------------------------------------------------
# Manage existing dataset (preview, caption, scan, delete, build)
# ---------------------------------------------------------------------------

from autotrain_sdk.dataset_manage import (
    dataset_stats as _ds_stats,
    scan_duplicates as _ds_scan,
    delete_all_images as _ds_del_imgs,
    resolve_dataset_path as _ds_resolve,
)
from autotrain_sdk.captioner import rename_and_caption_dataset as _rename_and_caption

def _manage_existing_dataset_smart():
    """Versión optimizada del manejo de datasets con búsqueda inteligente"""
    try:
        datasets_info = _get_datasets_cached()
        available = datasets_info['all']
    except Exception:
        # Fallback to original method
        from autotrain_sdk.paths import INPUT_DIR as _INPUT_DIR
        from autotrain_sdk.dataset_sources import list_external_datasets as _list_ext
        
        available = [p.name for p in _INPUT_DIR.glob("*") if p.is_dir()]
        available.extend(_list_ext().keys())
        available = sorted(set(available))
    
    if not available:
        console.print("[yellow]No datasets found in input/ or sources[/yellow]")
        _pause()
        return
    
    # Use smart search for large lists
    ds_name = _smart_search(available + ["Cancel"], "Select dataset")
    if not ds_name or ds_name == "Cancel":
        return
    
    # Add to recent datasets
    _add_recent_dataset(ds_name)
    
    # Continue with existing logic
    _manage_existing_dataset_impl(ds_name)

def _manage_existing_dataset_impl(ds_name):
    """Implementación común del manejo de datasets"""
    from autotrain_sdk.paths import BATCH_CONFIG_DIR
    
    # Registrar dataset como reciente cuando se gestiona
    _add_recent_dataset(ds_name)
    
    # Helper to print detailed stats & preview filenames
    def _print_stats(name: str):
        try:
            stats = _ds_stats(name)
        except Exception as e:
            console.print(f"[red]Error reading stats: {e}[/red]")
            return

        top_tok = ", ".join(f"{t}({c})" for t, c in stats["top_tokens"])
        console.rule(f"[bold yellow]{name}[/bold yellow]")
        console.print(
            f"[cyan]Images:[/cyan] {stats['images']}    "
            f"[cyan]Captions:[/cyan] {stats['txt']}    "
            f"[cyan]Avg res:[/cyan] {stats['avg_w']}×{stats['avg_h']}"
        )
        console.print(f"[cyan]Top tokens:[/cyan] {top_tok if top_tok else '-'}")

        # quick preview: first 8 filenames
        try:
            from pathlib import Path
            folder = _ds_resolve(name)
            if folder:
                imgs = sorted([p.name for p in folder.iterdir() if p.suffix.lower().lstrip('.') in SUPPORTED_IMAGE_EXTS])[:8]
                if imgs:
                    console.print(Panel("\n".join(imgs), title="Preview filenames", expand=False))
        except Exception:
            pass

    # Inner loop for actions on chosen dataset
    while True:
        _header("Dataset – Management")
        _print_stats(ds_name)

        action = q.select(
            "Choose action:",
            choices=[
                "Rename + Generate captions",
                "Scan duplicates / corrupt",
                "Delete ALL images",
                "Build / Update output structure",
                "Back",
            ],
        ).ask()

        if action == "Back" or action is None:
            break

        if action == "Rename + Generate captions":
            trigger = q.text("Trigger / base name:").ask()
            if not trigger:
                continue
            overwrite = q.confirm("Overwrite existing captions?", default=True).ask()
            max_tok = q.text("Max tokens (32-512):", default="128").ask()
            try:
                n_caps = _rename_and_caption(ds_name, trigger.strip(), overwrite=overwrite, max_new_tokens=int(max_tok))
                console.print(f"[green]{n_caps} caption(s) generated and images renamed.[/green]")
            except Exception as e:
                _error(str(e))
            _print_stats(ds_name)
            _pause()

        elif action == "Scan duplicates / corrupt":
            try:
                dups, corrs = _ds_scan(ds_name)
                console.print(f"[cyan]Duplicates:[/cyan] {dups}    [cyan]Corrupt:[/cyan] {corrs}")
            except Exception as e:
                _error(str(e))
            _print_stats(ds_name)
            _pause()

        elif action == "Delete ALL images":
            if not q.confirm("Are you sure? This will delete every image in the dataset.", default=False).ask():
                continue
            try:
                deleted = _ds_del_imgs(ds_name)
                console.print(f"[red]Deleted {deleted} images[/red]")
            except Exception as e:
                _error(str(e))
            _print_stats(ds_name)
            _pause()

        elif action == "Build / Update output structure":
            reps = q.text("Repetitions (times each image repeats):", default="30").ask()
            reso = q.text("Crop resolution WxH (blank = none):", default="").ask()
            try:
                from autotrain_sdk.gradio_app import cb_build_output_single as _cb_build_single

                msg = _cb_build_single(ds_name, int(reps or 30), res_str=(reso or None))
                console.print(f"[green]{msg}[/green]")
            except Exception as e:
                _error(str(e))
            _print_stats(ds_name)
            _pause()

        # loop continues showing updated stats

def _manage_existing_dataset():
    """Versión original mantenida para compatibilidad"""
    from autotrain_sdk.paths import INPUT_DIR as _INPUT_DIR
    from autotrain_sdk.dataset_sources import list_external_datasets as _list_ext
    
    # List available datasets (input/ plus external)
    available = [p.name for p in _INPUT_DIR.glob("*") if p.is_dir()]
    available.extend(_list_ext().keys())
    available = sorted(set(available))

    if not available:
        console.print("[yellow]No datasets found in input/ or sources[/yellow]")
        _pause()
        return

    ds_name = q.select("Select dataset:", choices=available + ["Cancel"]).ask()
    if not ds_name or ds_name == "Cancel":
        return
    
    _manage_existing_dataset_impl(ds_name)

# ---------------------------------------------------------------------------
# Bulk prepare multiple datasets (caption + output structure)
# ---------------------------------------------------------------------------

def _bulk_prepare_menu():
    """Wizard similar to Gradio *Bulk Prepare* tab."""

    from autotrain_sdk.paths import INPUT_DIR as _INPUT_DIR, OUTPUT_DIR

    # Collect dataset names (input + external)
    from autotrain_sdk.dataset_sources import list_external_datasets as _list_ext

    names = {p.name for p in _INPUT_DIR.glob("*") if p.is_dir()}
    names.update(_list_ext().keys())

    if not names:
        console.print("[yellow]No datasets available[/yellow]")
        _pause()
        return

    sel = q.checkbox(
        "Select datasets to process (space to mark):",
        choices=[q.Choice(n, checked=True) for n in sorted(names)],
    ).ask()

    if not sel:
        return

    # Check for existing caption files in selected datasets
    def has_txt_files(dataset_name):
        """Check if dataset has existing .txt caption files"""
        # Check in input/ directory first
        input_path = _INPUT_DIR / dataset_name
        if input_path.exists():
            txt_files = list(input_path.glob("*.txt"))
            if txt_files:
                return True, len(txt_files), "input/"
        
        # Check in external sources
        external_mapping = _list_ext()
        if dataset_name in external_mapping:
            external_path = Path(external_mapping[dataset_name])
            if external_path.exists():
                txt_files = list(external_path.glob("*.txt"))
                if txt_files:
                    return True, len(txt_files), str(external_path)
        
        return False, 0, ""

    # Analyze selected datasets for existing caption files
    datasets_with_captions = []
    total_txt_files = 0
    
    for dataset_name in sel:
        has_txt, count, location = has_txt_files(dataset_name)
        if has_txt:
            datasets_with_captions.append((dataset_name, count, location))
            total_txt_files += count

    # Improved caption generation options
    console.print("\n[bold blue]📝 Caption Generation Options[/bold blue]")
    console.print("[dim]Choose how to handle caption generation for all selected datasets:[/dim]")
    
    # Show information about existing caption files
    if datasets_with_captions:
        console.print(f"\n[yellow]⚠️  Found existing caption files:[/yellow]")
        for dataset_name, count, location in datasets_with_captions:
            console.print(f"  • [cyan]{dataset_name}[/cyan]: {count} .txt files in {location}")
        console.print(f"[dim]Total: {total_txt_files} caption files across {len(datasets_with_captions)} dataset(s)[/dim]")
    else:
        console.print(f"\n[green]✓ No existing caption files found in selected datasets[/green]")
    
    caption_mode = q.select(
        "Caption generation mode:",
        choices=[
            q.Choice("🎯 Generate with custom triggers (ask for each dataset)", "custom"),
            q.Choice("📁 Generate using dataset names as triggers", "auto"),
            q.Choice("⏭️  Skip caption generation entirely", "skip"),
        ],
        default="custom"
    ).ask()

    if caption_mode == "skip":
        generate_captions = False
        overwrite_capt = False
    else:
        generate_captions = True
        # Only ask about overwriting if there are existing caption files
        if datasets_with_captions:
            console.print(f"\n[yellow]Found {total_txt_files} existing caption files that could be overwritten.[/yellow]")
            overwrite_capt = q.confirm("Overwrite existing caption files?", default=True).ask()
        else:
            overwrite_capt = True  # No files to overwrite, so this doesn't matter
            console.print(f"[dim]No existing caption files to overwrite - proceeding with caption generation[/dim]")
    
    max_tok = int(q.text("Max tokens for captions (32-512):", default="128").ask() or 128) if generate_captions else 128

    # For each dataset ask optional trigger / reps / resolution
    rows = []
    for n in sel:
        console.print(f"\n[bold cyan]📂 {n}[/bold cyan]")
        
        if caption_mode == "custom":
            console.print(f"[dim]Leave trigger empty to use dataset name '{n}' as trigger[/dim]")
            trig = q.text("Custom trigger (blank = use dataset name):", default="").ask()
            # If empty, use dataset name as trigger
            final_trigger = trig.strip() if trig.strip() else n
        elif caption_mode == "auto":
            final_trigger = n
            console.print(f"[green]✓ Will use '{n}' as trigger[/green]")
        else:  # skip
            final_trigger = ""
        
        reps = int(q.text("Repetitions (times each image repeats):", default="30").ask() or 30)
        reso = q.text("Crop resolution WxH (blank = none):", default="").ask()
        rows.append((n, final_trigger, reps, reso.strip()))

    # Execute with progress bar
    from autotrain_sdk.gradio_app import cb_build_output_single as _cb_build_single

    def process_dataset(row_data):
        """Procesa un dataset individual - maneja tanto datasets locales como externos"""
        ds, trig, reps, reso = row_data
        try:
            # Verificar si el dataset está en input/ o necesita ser importado
            dest_path = _INPUT_DIR / ds
            
            if not dest_path.exists():
                # Dataset no está en input/, verificar si está en fuentes externas
                external_mapping = _list_ext()
                if ds in external_mapping:
                    # Importar dataset desde fuente externa
                    src_dir = external_mapping[ds]
                    shutil.copytree(src_dir, dest_path, dirs_exist_ok=True)
                    import_msg = f"imported from {src_dir}"
                else:
                    raise Exception(f"Dataset '{ds}' not found in input/ or external sources")
            else:
                import_msg = "found in input/"
            
            # Registrar dataset como reciente cuando se procesa en bulk
            _add_recent_dataset(ds)
            
            # Procesar dataset (ahora está garantizado en input/)
            cap_msg = ""
            if generate_captions and trig:
                n_caps = _rename_and_caption(ds, trig, overwrite=overwrite_capt, max_new_tokens=max_tok)
                cap_msg = f" · {n_caps} caption(s) with trigger '{trig}'"
            elif generate_captions and not trig:
                cap_msg = " · (captions skipped - no trigger provided)"
            else:
                cap_msg = " · (captions skipped by user choice)"

            msg_build = _cb_build_single(ds, reps, res_str=(reso or None))
            
            result_msg = f"{ds}: {msg_build}{cap_msg} ({import_msg})"
            return (True, result_msg)
        except Exception as e:
            error_msg = f"{ds}: ERROR – {e}"
            return (False, error_msg)
    
    console.print(f"\n[bold cyan]🚀 Starting bulk preparation of {len(rows)} datasets...[/bold cyan]")
    console.print()
    
    # Ejecutar con barra de progreso
    results = _progress_operation(
        items=rows,
        operation_func=process_dataset,
        title="📦 Processing datasets",
        item_name="dataset"
    )
    
    # Mostrar resultados
    console.print("\n[bold blue]📋 Bulk Prepare Results:[/bold blue]")
    success_count = 0
    for success, message in results:
        if success:
            console.print(f"[green]✓ {message}[/green]")
            success_count += 1
        else:
            console.print(f"[red]✗ {message}[/red]")
    
    console.print(f"\n[bold green]✅ Bulk prepare completed! {success_count}/{len(results)} datasets processed successfully.[/bold green]")
    _pause()

# ---------------------------------------------------------------------------
# Preset flows
# ---------------------------------------------------------------------------

def _presets_menu():
    from autotrain_sdk.paths import BATCH_CONFIG_DIR

    def _list_all_toml() -> List[Path]:
        return list(BATCH_CONFIG_DIR.glob("*/*.toml"))

    while True:
        _header_with_context("Presets", ["AutoTrainV2", "Presets"])
        
        # Mostrar información contextual
        _show_presets_overview()
        
        choice = q.select(
            "Choose action:",
            choices=[
                # 🔄 GENERATION
                q.Separator("╔═══════════════════╗"),
                q.Separator("║  🔄 GENERATION    ║"),
                q.Separator("╚═══════════════════╝"),
                q.Choice(" » 1. Regenerate all presets", "regen_all"),
                q.Choice("   2. Regenerate for specific dataset", "regen_dataset"),
                q.Separator("─────────────────────"),
                
                # 👁️ VIEWING
                q.Separator("╔═══════════════╗"),
                q.Separator("║  👁️ VIEWING   ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   3. List all presets", "list"),
                q.Choice("   4. View preset details", "view"),
                q.Separator("─────────────────"),
                
                # ✏️ EDITING
                q.Separator("╔═══════════════╗"),
                q.Separator("║  ✏️ EDITING   ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   5. Edit preset", "edit"),
                q.Choice("   6. Clone preset", "clone"),
                q.Separator("─────────────────"),
                
                q.Choice("   0. Back to main menu", "back"),
            ],
        ).ask()

        if choice == "regen_all":
            generate_presets()
            console.print("[green]Presets generated[/green]")
            _pause()

        elif choice == "regen_dataset":
            from autotrain_sdk.paths import INPUT_DIR

            available = [p.name for p in INPUT_DIR.glob("*") if p.is_dir()]
            if not available:
                console.print("[yellow]No folders in input/[/yellow]")
                _pause()
                continue
            dataset_name = q.select("Select dataset:", choices=available).ask()
            if not dataset_name:
                continue
            paths = generate_presets_for_dataset(dataset_name)
            if paths:
                console.print("[green]Presets generated:[/green]")
                for p in paths:
                    console.print(f" • {p}")
            else:
                console.print("[yellow]No presets generated (dataset nonexistent?).[/yellow]")
            _pause()

        elif choice == "list":
            files = _list_all_toml()
            if not files:
                console.print("[yellow]No presets found[/yellow]")
                _pause()
                continue

            # Build matrix: datasets × profiles
            profiles = ["Flux", "FluxLORA", "Nude"]
            datasets: set[str] = set()
            exists_map: dict[tuple[str, str], bool] = {}
            for p in files:
                prof = p.parent.name
                ds = p.stem
                datasets.add(ds)
                exists_map[(ds, prof)] = True

            datasets_sorted = sorted(datasets)
            table = Table("Dataset", *profiles)
            check = "✓"; cross = "✗"
            for ds in datasets_sorted:
                row = [ds]
                for prof in profiles:
                    row.append(check if exists_map.get((ds, prof)) else cross)
                table.add_row(*row)

            console.print(table)
            _pause()

        elif choice == "view":
            files = _list_all_toml()
            if not files:
                console.print("[yellow]No presets available[/yellow]")
                _pause()
                continue
            selected = q.select("Select preset:", choices=[str(p) for p in files]).ask()
            if selected:
                cfg = load_config(Path(selected))
                table = Table("Key", "Value", title=selected)
                for k, v in cfg.items():
                    table.add_row(str(k), str(v))
                console.print(table)
                _pause()

        elif choice == "edit":
            files = _list_all_toml()
            if not files:
                console.print("[yellow]No presets available[/yellow]")
                _pause()
                continue
            selected = q.select("Select preset to edit:", choices=[str(p) for p in files]).ask()
            if selected:
                _edit_preset_interactive(Path(selected))

        elif choice == "clone":
            files = _list_all_toml()
            if not files:
                console.print("[yellow]No presets to clone[/yellow]")
                _pause()
                continue
            src_path_str = q.select("Select source preset:", choices=[str(p) for p in files]).ask()
            if not src_path_str:
                continue
            src_path = Path(src_path_str)
            new_name = q.text("New file name (without .toml):").ask()
            if not new_name:
                continue
            dest_path = src_path.parent / f"{new_name}.toml"
            if dest_path.exists():
                _error("Destination file already exists.")
                continue

            shutil.copy2(src_path, dest_path)
            console.print(f"[green]Cloned to {dest_path}[/green]")
            _pause()

        else:
            break

# ---------------------------------------------------------------------------
# Training flow
# ---------------------------------------------------------------------------

def _training_menu():
    """
    TRAINING MENU: Pure Setup & Configuration
    - No job management here
    - Focus on training preparation and setup
    """
    from autotrain_sdk.paths import BATCH_CONFIG_DIR

    while True:
        _header_with_context("Training", ["AutoTrainV2", "Training"])
        
        # Show available presets summary
        profiles = ["Flux", "FluxLORA", "Nude"]
        profile_counts = []
        
        for profile in profiles:
            presets = list((BATCH_CONFIG_DIR / profile).glob("*.toml"))
            count = len(presets)
            if count > 0:
                profile_counts.append(f"{profile}: {count}")
        
        if profile_counts:
            console.print(f"📋 Available presets: {', '.join(profile_counts)}")
            console.print()
        
        action = q.select(
            "Choose training action:",
            choices=[
                # 🚀 TRAINING MODES
                q.Separator("╔═══════════════════════╗"),
                q.Separator("║  🚀 TRAINING MODES    ║"),
                q.Separator("╚═══════════════════════╝"),
                q.Choice(" » 1. Single dataset training", "single_mode"),
                q.Choice("   2. Batch training (multiple datasets)", "batch_mode"),
                q.Separator("─────────────────────────"),
                
                # 📊 ANALYSIS
                q.Separator("╔═══════════════╗"),
                q.Separator("║  📊 ANALYSIS  ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   3. Dataset analysis", "dataset_analysis"),
                q.Choice("   4. Training time estimation", "training_estimation"),
                q.Separator("─────────────────"),
                
                q.Choice("   0. Back to main menu", "back"),
            ],
        ).ask()
        
        if action == "back":
            break
        elif action == "single_mode":
            _training_advanced_setup()
        elif action == "batch_mode":
            _training_batch_menu()
        elif action == "dataset_analysis":
            _training_info_menu()
        elif action == "training_estimation":
            _training_estimation_menu()

def _jobs_menu():
    """
    JOBS MENU: Pure Management & Monitoring
    - No training setup here
    - Focus on job lifecycle management
    - Auto-switches to optimized view for large queues (>100 jobs)
    """
    
    # Check if we should use optimized view for large queues
    all_jobs = _JOB_MANAGER.list_jobs()
    jobs_count = len(all_jobs)
    
    # Auto-switch to optimized view for large queues
    if jobs_count > 100:
        console.print(f"[yellow]⚡ Large queue detected ({jobs_count} jobs) - switching to optimized view[/yellow]")
        console.print("[dim]This provides better performance and navigation for large job queues[/dim]")
        console.print()
        _jobs_menu_optimized()
        return
    
    while True:
        _header_with_context("Job Management", ["AutoTrainV2", "Jobs"])
        
        # Show job summary
        _show_job_summary()
        
        action = q.select(
            "Choose job action:",
            choices=[
                # 📋 QUEUE MANAGEMENT
                q.Separator("╔═══════════════════════╗"),
                q.Separator("║  📋 QUEUE MANAGEMENT  ║"),
                q.Separator("╚═══════════════════════╝"),
                q.Choice(" » 1. View training queue", "view_queue"),
                q.Choice("   2. Cancel/Remove jobs", "cancel_remove"),
                q.Separator("─────────────────────────"),
                
                # 🔍 MONITORING
                q.Separator("╔═══════════════════╗"),
                q.Separator("║  🔍 MONITORING    ║"),
                q.Separator("╚═══════════════════╝"),
                q.Choice("   3. Live monitor", "live_monitor"),
                q.Choice("   4. Job details", "job_details"),
                q.Separator("─────────────────────"),
                
                # 📈 ANALYSIS
                q.Separator("╔═══════════════╗"),
                q.Separator("║  📈 ANALYSIS  ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   5. Training history", "training_history"),
                q.Separator("─────────────────"),
                
                # ⚡ OPTIMIZED VIEW
                q.Choice("   6. Optimized view (for large queues)", "optimized"),
                q.Choice("   0. Back to main menu", "back"),
            ],
        ).ask()
        
        if action == "back":
            break
        elif action == "view_queue":
            _view_queue_compact()
        elif action == "cancel_remove":
            _cancel_remove_jobs()
        elif action == "live_monitor":
            _training_monitor_menu()
        elif action == "job_details":
            _job_details_menu()
        elif action == "training_history":
            _training_history_menu()
        elif action == "optimized":
            _jobs_menu_optimized()

# ---------------------------------------------------------------------------
# OPTION A: Compact List View Implementation
# ---------------------------------------------------------------------------

def _create_compact_job_table() -> Table:
    """Create a compact job table (Option A implementation)."""
    jobs = _JOB_MANAGER.list_jobs()
    
    table = Table(show_header=True, header_style="bold magenta", box=None, title="📋 Job Queue")
    table.add_column("Job", style="cyan", width=8)
    table.add_column("Dataset", style="blue", width=12)
    table.add_column("Status", style="bold", width=12)
    table.add_column("Progress", style="green", width=15)
    table.add_column("ETA", style="yellow", width=8)
    
    if not jobs:
        table.add_row("—", "—", "—", "No jobs in queue", "—")
        return table
    
    for job in jobs:
        # Format status with emoji
        status_text = _format_job_status_menu(job.status)
        
        # Format progress
        if job.status == JobStatus.RUNNING and job.progress_str:
            progress_text = f"{job.percent:.0f}% ({job.current_step}/{job.total_steps})"
        elif job.status == JobStatus.PENDING:
            # Calculate queue position
            pending_jobs = [j for j in jobs if j.status == JobStatus.PENDING]
            position = next((i+1 for i, j in enumerate(pending_jobs) if j.id == job.id), 0)
            progress_text = f"Queued #{position}"
        elif job.status == JobStatus.DONE:
            progress_text = "100%"
        elif job.status == JobStatus.FAILED:
            progress_text = "Failed"
        elif job.status == JobStatus.CANCELED:
            progress_text = "Canceled"
        else:
            progress_text = "—"
        
        # Format ETA
        if job.status == JobStatus.RUNNING and job.eta:
            eta_text = job.eta
        elif job.status == JobStatus.PENDING:
            eta_text = "~" + str(position * 2) + "h"  # Rough estimate
        elif job.status == JobStatus.DONE:
            eta_text = "Complete"
        else:
            eta_text = "—"
        
        table.add_row(
            job.id,
            job.dataset[:12],  # Truncate long dataset names
            status_text,
            progress_text,
            eta_text
        )
    
    return table

def _show_job_summary():
    """Show a brief job summary at the top of jobs menu."""
    jobs = _JOB_MANAGER.list_jobs()
    
    if not jobs:
        console.print("[dim]📋 No jobs in system[/dim]")
        console.print()
        return
    
    # Count by status
    status_counts = {}
    for job in jobs:
        status = job.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Format summary
    summary_parts = []
    if status_counts.get("running", 0) > 0:
        summary_parts.append(f"🔄 {status_counts['running']} running")
    if status_counts.get("pending", 0) > 0:
        summary_parts.append(f"⏸️ {status_counts['pending']} pending")
    if status_counts.get("done", 0) > 0:
        summary_parts.append(f"✅ {status_counts['done']} completed")
    if status_counts.get("failed", 0) > 0:
        summary_parts.append(f"❌ {status_counts['failed']} failed")
    
    if summary_parts:
        console.print(f"[dim]📊 Queue: {' | '.join(summary_parts)}[/dim]")
        console.print()

def _view_queue_compact():
    """View job queue using compact list format with smart optimization detection."""
    _header_with_context("Job Queue", ["AutoTrainV2", "Jobs", "Queue"])
    
    # Check if optimized view should be used
    all_jobs = _JOB_MANAGER.list_jobs()
    
    if _should_use_optimized_view(all_jobs):
        _show_queue_performance_warning(len(all_jobs))
        
        # Offer to switch to optimized view
        use_optimized = q.select(
            "Large queue detected. Choose view:",
            choices=[
                q.Choice("Use optimized view (recommended)", True),
                q.Choice("Continue with legacy view", False),
            ]
        ).ask()
        
        if use_optimized:
            _jobs_menu_optimized()
            return
    
    table = _create_compact_job_table()
    console.print(table)
    console.print()
    
    # Enhanced quick actions based on queue size
    if len(all_jobs) > 50:
        console.print("[dim]Quick actions: [r]efresh, [d]etails, [c]ancel, [o]ptimized view, [q]uit[/dim]")
    else:
        console.print("[dim]Quick actions: [r]efresh, [d]etails, [c]ancel, [q]uit[/dim]")
    
    while True:
        action = q.text("Action (or Enter to back):").ask()
        if not action or action.lower() == 'q':
            break
        elif action.lower() == 'r':
            console.clear()
            _header_with_context("Job Queue", ["AutoTrainV2", "Jobs", "Queue"])
            
            # Re-check for optimization on refresh
            current_jobs = _JOB_MANAGER.list_jobs()
            if _should_use_optimized_view(current_jobs) and len(current_jobs) != len(all_jobs):
                console.print(f"[yellow]Queue size changed to {len(current_jobs)} jobs - consider optimized view[/yellow]")
                console.print()
            
            table = _create_compact_job_table()
            console.print(table)
            console.print()
            
            if len(current_jobs) > 50:
                console.print("[dim]Quick actions: [r]efresh, [d]etails, [c]ancel, [o]ptimized view, [q]uit[/dim]")
            else:
                console.print("[dim]Quick actions: [r]efresh, [d]etails, [c]ancel, [q]uit[/dim]")
                
        elif action.lower() == 'd':
            _job_details_menu()
        elif action.lower() == 'c':
            _cancel_remove_jobs()
        elif action.lower() == 'o' and len(all_jobs) > 50:
            _jobs_menu_optimized()
            break
        else:
            console.print("[yellow]Unknown action. Use r/d/c/q" + ("/o" if len(all_jobs) > 50 else "") + "[/yellow]")

# ---------------------------------------------------------------------------
# TRAINING MENU FUNCTIONS (Setup & Configuration)
# ---------------------------------------------------------------------------

def _training_advanced_setup():
    """Single mode training setup with all options."""
    _header_with_context("Single Mode Training", ["AutoTrainV2", "Training", "Single Mode"])
    
    # This redirects to the original training logic but cleaned up
    _training_original_logic()

def _training_estimation_menu():
    """Dedicated training estimation interface."""
    _header_with_context("Training Estimation", ["AutoTrainV2", "Training", "Estimation"])
    
    # Get available datasets
    try:
        datasets_info = _get_datasets_cached()
        available = datasets_info['all']
    except Exception:
        from autotrain_sdk.paths import INPUT_DIR
        available = [p.name for p in INPUT_DIR.glob("*") if p.is_dir()]
    
    if not available:
        console.print("[yellow]No datasets found[/yellow]")
        _pause()
        return
    
    # Select dataset and profile
    dataset = q.select("Select dataset:", choices=available + ["Cancel"]).ask()
    if not dataset or dataset == "Cancel":
        return
    
    profile = q.select("Select profile:", choices=["Flux", "FluxLORA", "Nude"], default="FluxLORA").ask()
    if not profile:
        return
    
    # Show estimation
    console.clear()
    _header_with_context(f"Training Estimation: {dataset}", ["AutoTrainV2", "Training", "Estimation"])
    
    info_table = _create_dataset_info_table_menu(dataset, profile)
    console.print(info_table)
    console.print()
    
    estimation_table = _create_training_estimation_table_menu(dataset, profile)
    console.print(estimation_table)
    console.print()
    
    _pause()

# ---------------------------------------------------------------------------
# JOBS MENU FUNCTIONS (Management & Monitoring)
# ---------------------------------------------------------------------------

def _cancel_remove_jobs():
    """Cancel or remove jobs from queue."""
    _header_with_context("Cancel/Remove Jobs", ["AutoTrainV2", "Jobs", "Cancel"])
    
    jobs = _JOB_MANAGER.list_jobs()
    if not jobs:
        console.print("[yellow]No jobs in queue[/yellow]")
        _pause()
        return
    
    # Show current queue
    table = _create_compact_job_table()
    console.print(table)
    console.print()
    
    # Select action
    action = q.select(
        "Choose action:",
        choices=[
            "Cancel running job",
            "Remove completed job", 
            "Emergency stop all",
            "Back"
        ]
    ).ask()
    
    if action == "Back" or not action:
        return
    elif action == "Cancel running job":
        running_jobs = [j for j in jobs if j.status in {JobStatus.PENDING, JobStatus.RUNNING}]
        if not running_jobs:
            console.print("[yellow]No running jobs to cancel[/yellow]")
            _pause()
            return
        
        job_choices = [f"{j.id} ({j.dataset}, {j.status.value})" for j in running_jobs]
        selected = q.select("Select job to cancel:", choices=job_choices + ["Cancel"]).ask()
        if not selected or selected == "Cancel":
            return
        
        job_id = selected.split(" ")[0]
        if q.confirm(f"Cancel job {job_id}?", default=False).ask():
            _JOB_MANAGER.cancel(job_id)
            console.print(f"[green]✅ Job {job_id} canceled[/green]")
            _pause()
    
    elif action == "Remove completed job":
        completed_jobs = [j for j in jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
        if not completed_jobs:
            console.print("[yellow]No completed jobs to remove[/yellow]")
            _pause()
            return
        
        job_choices = [f"{j.id} ({j.dataset}, {j.status.value})" for j in completed_jobs]
        selected = q.select("Select job to remove:", choices=job_choices + ["Cancel"]).ask()
        if not selected or selected == "Cancel":
            return
        
        job_id = selected.split(" ")[0]
        if q.confirm(f"Remove job {job_id} from queue?", default=True).ask():
            try:
                if _JOB_MANAGER.remove_job(job_id):
                    console.print(f"[green]✅ Job {job_id} removed[/green]")
                else:
                    console.print(f"[yellow]⚠️ Could not remove job {job_id}[/yellow]")
            except AttributeError:
                console.print(f"[yellow]⚠️ remove_job not implemented[/yellow]")
            _pause()
    
    elif action == "Emergency stop all":
        running_jobs = [j for j in jobs if j.status in {JobStatus.PENDING, JobStatus.RUNNING}]
        if not running_jobs:
            console.print("[yellow]No running jobs to stop[/yellow]")
            _pause()
            return
        
        console.print(f"[red]⚠️ This will stop {len(running_jobs)} running job(s)[/red]")
        if q.confirm("Emergency stop all jobs?", default=False).ask():
            stopped = 0
            for job in running_jobs:
                _JOB_MANAGER.cancel(job.id)
                stopped += 1
            console.print(f"[green]✅ Emergency stop initiated for {stopped} jobs[/green]")
            _pause()

def _clean_completed_jobs():
    """Clean completed jobs from queue."""
    _header_with_context("Clean Completed Jobs", ["AutoTrainV2", "Jobs", "Clean"])
    
    jobs = _JOB_MANAGER.list_jobs()
    completed_jobs = [j for j in jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
    
    if not completed_jobs:
        console.print("[yellow]No completed jobs to clean[/yellow]")
        _pause()
        return
    
    console.print(f"[cyan]Found {len(completed_jobs)} completed job(s):[/cyan]")
    for job in completed_jobs:
        console.print(f"  • {job.id} ({job.dataset}) - {job.status.value}")
    console.print()
    
    if q.confirm(f"Remove {len(completed_jobs)} completed job(s) from queue?", default=True).ask():
        removed = 0
        for job in completed_jobs:
            try:
                if _JOB_MANAGER.remove_job(job.id):
                    removed += 1
            except AttributeError:
                console.print(f"[yellow]⚠️ remove_job not implemented - completed jobs will remain in queue[/yellow]")
                break
        console.print(f"[green]✅ Cleaned {removed} job(s) from queue[/green]")
    
    _pause()

def _job_details_menu():
    """Show detailed information about jobs."""
    _header_with_context("Job Details", ["AutoTrainV2", "Jobs", "Details"])
    
    jobs = _JOB_MANAGER.list_jobs()
    if not jobs:
        console.print("[yellow]No jobs in queue[/yellow]")
        _pause()
        return
    
    # Show compact list first
    table = _create_compact_job_table()
    console.print(table)
    console.print()
    
    # Select job for details
    job_choices = [f"{j.id} ({j.dataset}, {j.status.value})" for j in jobs]
    selected = q.select("Select job for details:", choices=job_choices + ["Cancel"]).ask()
    if not selected or selected == "Cancel":
        return
    
    job_id = selected.split(" ")[0]
    job = next((j for j in jobs if j.id == job_id), None)
    if job:
        _show_job_details_menu(job)



# ---------------------------------------------------------------------------
# HELPER FUNCTION FOR ORIGINAL TRAINING LOGIC
# ---------------------------------------------------------------------------

def _training_original_logic():
    """Original training logic for advanced setup."""
    from autotrain_sdk.paths import BATCH_CONFIG_DIR

    # Find available presets
    profiles = ["Flux", "FluxLORA", "Nude"]
    action = q.select("Choose training profile:", 
                     choices=[q.Choice(f"{p}", p.lower()) for p in profiles] + [q.Choice("Cancel", "cancel")]).ask()
    
    if action == "cancel":
        return
    
    profile_map = {"flux": "Flux", "fluxlora": "FluxLORA", "nude": "Nude"}
    profile = profile_map.get(action)
    
    if not profile:
        return
    
    # Find available presets
    presets = list((BATCH_CONFIG_DIR / profile).glob("*.toml"))
    if not presets:
        console.print(f"[red]No presets in BatchConfig/{profile}[/red]")
        _pause()
        return

    # Multi-select datasets
    choices = [q.Choice(f"{p.stem} ({p.name})", value=str(p)) for p in presets]
    sel_paths = q.checkbox("Select one or more datasets (space to mark):", choices=choices).ask()
    if not sel_paths:
        return

    gpu_ids = _select_gpus()

    def enqueue_job(path_str):
        """Encola un job individual"""
        try:
            preset_path = Path(path_str)
            dataset_name = preset_path.stem
            
            # Registrar dataset como reciente cuando se entrena
            _add_recent_dataset(dataset_name)
            
            from autotrain_sdk.paths import compute_run_dir
            run_dir = compute_run_dir(dataset_name, profile)
            job = Job(dataset_name, profile, preset_path, run_dir, gpu_ids=gpu_ids)
            _JOB_MANAGER.enqueue(job)
            
            return (True, f"{dataset_name} → Job {job.id} (Output: {run_dir})")
        except Exception as e:
            return (False, f"{Path(path_str).stem} → Error: {str(e)}")
    
    console.print(f"\n[bold cyan]🚀 Enqueuing {len(sel_paths)} training jobs...[/bold cyan]")
    console.print()
    
    # Ejecutar con barra de progreso
    results = _progress_operation(
        items=sel_paths,
        operation_func=enqueue_job,
        title="🎯 Enqueuing training jobs",
        item_name="job"
    )
    
    # Mostrar resultados
    console.print("\n[bold blue]📋 Job Queue Results:[/bold blue]")
    success_count = 0
    for success, message in results:
        if success:
            console.print(f"[green]✓ {message}[/green]")
            success_count += 1
        else:
            console.print(f"[red]✗ {message}[/red]")
    
    console.print(f"\n[bold green]✅ Training queue completed! {success_count}/{len(results)} jobs enqueued successfully.[/bold green]")
    _pause()

# ---------------------------------------------------------------------------
# REMOVE DUPLICATED FUNCTIONS (Clean up old implementations)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# QuickTrain (one-click) CLI flow
# ---------------------------------------------------------------------------

def _quicktrain_menu():
    """Minimal wizard: dataset path -> prepare -> enqueue job."""

    _header_with_context("QuickTrain", ["AutoTrainV2", "QuickTrain"])

    from autotrain_sdk.paths import INPUT_DIR, BATCH_CONFIG_DIR, compute_run_dir
    from autotrain_sdk import dataset_sources as ds_src

    console.print("One-click training with automatic setup")
    console.print()

    # Step 1: Dataset selection
    existing_names = {p.name for p in INPUT_DIR.glob("*") if p.is_dir()}
    external_map = ds_src.list_external_datasets()
    existing_names.update(external_map.keys())
    choices_ds = sorted(existing_names)
    choices_ds.append("<Provide absolute path manually>")

    console.print("[bold]Step 1: Select Dataset[/bold]")
    sel_name = q.select("Choose dataset source:", choices=choices_ds).ask()
    if sel_name is None:
        return

    if sel_name == "<Provide absolute path manually>":
        ds_path = q.path("Dataset folder path (absolute):").ask()
        if not ds_path:
            return
        src = Path(ds_path)
    else:
        # resolve path automatically
        p_in = INPUT_DIR / sel_name
        if p_in.exists():
            src = p_in
        else:
            src = Path(external_map.get(sel_name, ""))
    
    if not src.exists():
        _error(f"Path not found: {src}")
        return

    if not src.is_dir():
        _error("Selected path is not a directory")
        return

    dataset_name = src.name
    
    # Registrar dataset como reciente cuando se usa en QuickTrain
    _add_recent_dataset(dataset_name)

    # Step 2: Training configuration
    console.print(f"\n[bold]Step 2: Training Configuration[/bold]")
    console.print(f"Dataset: [cyan]{dataset_name}[/cyan]")
    
    # Use user preferences for defaults
    user_config = _load_user_config()
    default_profile = user_config.get("default_profile", "FluxLORA")
    
    remote = q.confirm("Store output in remote path?", default=False).ask()

    mode = q.select("Training mode:", 
                   choices=[
                       q.Choice("Flux - Full model training", "Flux"),
                       q.Choice("FluxLORA - LoRA fine-tuning (recommended)", "FluxLORA"),
                       q.Choice("Nude - Specialized training", "Nude")
                   ], 
                   default=default_profile).ask()
    if mode is None:
        return

    # Step 3: Advanced options
    console.print(f"\n[bold]Step 3: Advanced Options (Optional)[/bold]")
    show_advanced = q.confirm("Configure advanced training parameters?", default=False).ask()
    
    epochs_val = batch_val = lr_val = ""
    if show_advanced:
        epochs_val = q.text("Max epochs (blank = preset default):", default="").ask()
        batch_val = q.text("Train batch size (blank = preset default):", default="").ask()
        lr_val = q.text("Learning rate (blank = preset default):", default="").ask()

    gpu_ids = _select_gpus()

    # Step 4: Processing
    console.print(f"\n[bold]Step 4: Processing[/bold]")
    
    # Definir pasos del QuickTrain
    steps = []
    
    # Step 1: Copy dataset if needed
    dest = INPUT_DIR / dataset_name
    if not dest.exists():
        def copy_dataset():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return f"Dataset copied to {dest}"
        steps.append(("📁 Copying dataset", copy_dataset))
    
    # Step 2: Set remote flag
    def set_remote_flag():
        if remote:
            os.environ["AUTO_REMOTE_DIRECT"] = "1"
        else:
            os.environ.pop("AUTO_REMOTE_DIRECT", None)
        return f"Remote mode: {'enabled' if remote else 'disabled'}"
    steps.append(("🌐 Configuring remote settings", set_remote_flag))
    
    # Step 3: Prepare output structure
    def prepare_output():
        populate_output_structure_single(dataset_name)
        return "Output structure prepared"
    steps.append(("🔄 Preparing output structure", prepare_output))
    
    # Step 4: Generate presets
    def generate_presets():
        generate_presets_for_dataset(dataset_name)
        return "Presets generated"
    steps.append(("📋 Generating presets", generate_presets))
    
    # Step 5: Setup directories
    def setup_directories():
        run_dir = compute_run_dir(dataset_name, mode)
        run_dir.mkdir(parents=True, exist_ok=True)
        return f"Run directory: {run_dir}"
    steps.append(("📁 Creating run directory", setup_directories))
    
    # Step 6: Configure training parameters
    def configure_training():
        preset_dir = BATCH_CONFIG_DIR / mode
        cfg_path = preset_dir / f"{dataset_name}.toml"
        if not cfg_path.exists():
            raise Exception("Preset not found after generation")
        
        run_dir = compute_run_dir(dataset_name, mode)
        cfg = toml.load(cfg_path)
        cfg["output_dir"] = str(run_dir)
        cfg["logging_dir"] = str(run_dir/"log")
        
        if epochs_val.strip():
            cfg["max_train_epochs"] = int(epochs_val)
        if batch_val.strip():
            cfg["train_batch_size"] = int(batch_val)
        if lr_val.strip():
            cfg["learning_rate"] = float(lr_val)
        
        patched = run_dir/"config.toml"
        with patched.open("w",encoding="utf-8") as f:
            toml.dump(cfg,f)
        
        return f"Config saved to {patched}"
    steps.append(("⚙️ Configuring training parameters", configure_training))
    
    # Step 7: Enqueue job
    def enqueue_training_job():
        run_dir = compute_run_dir(dataset_name, mode)
        patched = run_dir/"config.toml"
        job = Job(dataset_name, mode, patched, run_dir, gpu_ids=gpu_ids)
        _JOB_MANAGER.enqueue(job)
        return f"Job {job.id} enqueued"
    steps.append(("🎯 Enqueuing training job", enqueue_training_job))
    
    console.print(f"[bold cyan]🚀 Starting QuickTrain setup with {len(steps)} steps...[/bold cyan]")
    console.print()
    
    # Ejecutar pasos con barra de progreso
    results = _progress_with_steps(steps, "⚡ QuickTrain Setup")
    
    # Verificar resultados
    failed_steps = [i for i, (success, _) in enumerate(results) if not success]
    
    if failed_steps:
        console.print("\n[red]❌ QuickTrain failed at the following steps:[/red]")
        for i in failed_steps:
            step_name, _ = steps[i]
            _, error_msg = results[i]
            console.print(f"[red]  • {step_name}: {error_msg}[/red]")
        _pause()
        return
    
    # Obtener información del job final
    final_result = results[-1][1]  # Resultado del último paso (enqueue job)
    
    console.print(f"\n[green]✅ QuickTrain Complete![/green]")
    console.print(f"Job {job.id} has been queued for training")
    console.print(f"Output directory: [cyan]{run_dir}[/cyan]")
    console.print(f"Monitor progress in: [bold]5. Job queue → 4. View live progress[/bold]")
    _pause()

# ---------------------------------------------------------------------------
# Web UI flow
# ---------------------------------------------------------------------------

def _web_menu():
    _header_with_context("Web Interface", ["AutoTrainV2", "Web UI"])
    console.print("[dim]🌐 Opening Gradio web interface...[/dim]")
    console.print()
    
    if q.confirm("Share public link (accessible from internet)?", default=False).ask():
        console.print("[yellow]⚠️  Creating public link - this may take a moment...[/yellow]")
        launch_web(share=True)
    else:
        console.print("[cyan]🔒 Local interface only[/cyan]")
        launch_web()

# ---------------------------------------------------------------------------
# Logs flow
# ---------------------------------------------------------------------------

def _logs_menu():
    LOGS_DIR.mkdir(exist_ok=True)

    def _list_logs():
        return sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    while True:
        _header_with_context("Log Viewer", ["AutoTrainV2", "Logs"])
        
        files = _list_logs()
        console.print(f"[dim]📄 Found {len(files)} log files[/dim]")
        console.print()
        
        choices = [str(p.name) for p in files]
        choice = q.select(
            "Available logs (recent first):",
            choices=choices + [q.Separator(), "Refresh", "Back"],
        ).ask()

        if choice in {None, "Back"}:
            return
        if choice == "Refresh":
            continue

        # User selected a log
        log_path = LOGS_DIR / choice
        action = q.select(
            f"Log '{choice}'",
            choices=["View full", "View last 50 lines", "Follow live (tail)", "Back"],
        ).ask()

        if action == "Back" or action is None:
            continue

        if action == "View full":
            _print_log(log_path)
        elif action == "View last 50 lines":
            _print_log(log_path, tail=50)
        else:  # tail live
            console.rule(f"Following {choice} (Ctrl-C to exit)")
            try:
                _follow_log(log_path)
            except KeyboardInterrupt:
                console.print("[yellow]\nFollowing stopped[/yellow]")
        _pause()

def _print_log(path: Path, tail: int | None = None):
    """Print full log or last *tail* lines."""

    from collections import deque

    if tail is not None:
        dq: deque[str] = deque(maxlen=tail)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                dq.append(line.rstrip())
        console.print("\n".join(dq))
    else:
        # Stream file in chunks to reduce memory usage
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            chunk: list[str] = []
            for idx, line in enumerate(f, 1):
                chunk.append(line.rstrip())
                if idx % 500 == 0:
                    console.print("\n".join(chunk))
                    chunk.clear()
            if chunk:
                console.print("\n".join(chunk))

def _follow_log(path: Path):
    """Imitates 'tail -f' showing new lines in real time."""

    import time
    import os

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)  # start at end
        last_size = path.stat().st_size
        while True:
            line = f.readline()
            if line:
                console.print(line.rstrip())
                last_size = f.tell()
                continue

            # EOF: wait and detect truncation
            time.sleep(0.5)
            try:
                new_size = path.stat().st_size
            except FileNotFoundError:
                continue  # file might be rotated; wait

            if new_size < last_size:
                # File truncated (new run). Rewind
                f.seek(0)
                last_size = 0

# ---------------------------------------------------------------------------
# Preset editor
# ---------------------------------------------------------------------------

from pydantic import ValidationError
from autotrain_sdk.config_models import TrainingConfig

def _coerce_value(value: str):
    """Tries to convert strings to int, float or bool when possible"""

    lowers = value.lower()
    if lowers in {"true", "false"}:
        return lowers == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

def _edit_preset_interactive(path: Path):
    """Allows user to edit keys of a preset."""

    cfg = load_config(path)

    while True:
        keys = sorted(cfg.keys())
        choice = q.select(
            f"Edit preset: {path.name}",
            choices=keys + [q.Separator(), "Add new key", "Save and exit", "Cancel changes"],
        ).ask()

        if choice is None:
            return

        if choice == "Add new key":
            new_key = q.text("Name of new key:").ask()
            if not new_key:
                continue
            if new_key in cfg:
                _error("Key already exists.")
                continue
            raw_val = q.text("Value:").ask()
            cfg[new_key] = _coerce_value(raw_val)

        elif choice == "Save and exit":
            # Validate before saving
            try:
                TrainingConfig(**cfg)
            except ValidationError as e:
                _error(f"Validation failed:\n{e}")
                continue
            save_config(path, cfg)
            console.print("[green]Preset saved[/green]")
            _pause()
            return

        elif choice == "Cancel changes":
            console.print("[yellow]Changes discarded[/yellow]")
            _pause()
            return

        else:  # edit existing key
            old_val = cfg.get(choice)
            new_raw = q.text(f"New value for '{choice}' (current: {old_val}):").ask()
            if new_raw is None:
                continue
            cfg[choice] = _coerce_value(new_raw)

# ---------------------------------------------------------------------------
# Jobs queue flow
# ---------------------------------------------------------------------------

_JOB_MANAGER = JobManager()

# ---------------------------------------------------------------------------
# GPU selection helper
# ---------------------------------------------------------------------------

def _select_gpus() -> str | None:
    """Returns a string '0,1' with selected GPUs or None to use all."""

    try:
        import GPUtil  # type: ignore

        gpus = GPUtil.getGPUs()
        if not gpus:
            return None
        choices = [f"{gpu.id} – {gpu.memoryFree}MB free" for gpu in gpus]
        selected = q.checkbox(
            "Select GPU(s) (space to mark):",
            choices=[q.Choice(ch, checked=(i == 0)) for i, ch in enumerate(choices)],
        ).ask()
        ids = [c.split(" –", 1)[0] for c in selected]
        return ",".join(ids) if ids else None
    except Exception:
        # fallback: read env var
        env_gpu = os.getenv("CUDA_VISIBLE_DEVICES")
        return env_gpu

# ---------------------------------------------------------------------------
# External dataset sources helpers
# ---------------------------------------------------------------------------

from autotrain_sdk import dataset_sources as ds_src


def _sources_menu():
    while True:
        _header_with_context("External Sources", ["AutoTrainV2", "Datasets", "External Sources"])
        
        current = ds_src.load_sources()
        
        # Show currently configured source paths
        if current:
            console.print(f"[bold]📁 Configured source paths ({len(current)}):[/bold]")
            for i, path in enumerate(current, 1):
                console.print(f"  {i}. [cyan]{path}[/cyan]")
            
            # Show how many datasets are found in these sources
            try:
                external_datasets = ds_src.list_external_datasets()
                console.print(f"\n[dim]Found {len(external_datasets)} external datasets across all sources[/dim]")
            except Exception:
                pass
            console.print()
        else:
            console.print("[yellow]No external source paths configured[/yellow]")
            console.print("[dim]Add folder paths containing datasets to scan for external sources[/dim]")
            console.print()
        
        choices = [str(p) for p in current]
        action = q.select(
            "Choose action:",
            choices=[
                q.Choice("1. Add folder", "add"),
                q.Choice("2. Remove folder", "remove"),
                q.Separator(),
                q.Choice("0. Back", "back"),
            ],
        ).ask()

        if action in {None, "back"}:
            return

        if action == "add":
            path = q.path("Path to folder with datasets:").ask()
            if path:
                ds_src.add_source(path)
                console.print(f"[green]Added {path}[/green]")
            _pause()

        elif action == "remove":
            if not choices:
                console.print("[yellow]No folders registered[/yellow]")
                _pause()
                continue
            target = q.select("Select folder to remove:", choices=choices).ask()
            if target:
                ds_src.remove_source(target)
                console.print(f"[red]Removed {target}[/red]")
            _pause()


def _import_external_dataset():
    mapping = ds_src.list_external_datasets()
    if not mapping:
        console.print("[yellow]No external datasets found[/yellow]")
        _pause()
        return
    # exclude those already present in input/
    from autotrain_sdk.paths import INPUT_DIR

    existing = {p.name for p in INPUT_DIR.glob("*") if p.is_dir()}
    choices = [name for name in mapping.keys() if name not in existing]
    if not choices:
        console.print("[yellow]All external datasets already exist in input/[/yellow]")
        _pause()
        return
    selected = q.checkbox(
        "Select datasets to import (space to mark):",
        choices=[q.Choice(n, checked=True) for n in choices],
    ).ask()
    if not selected:
        return

    for name in selected:
        src_dir = mapping[name]
        dest_dir = INPUT_DIR / name
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        console.print(f"[green]Imported {name}[/green]")
    _pause()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():  # noqa: D401 (simple verb)
    """Executes main menu with improved navigation and grouping."""
    # Initialize environment variables from config file at startup
    from .utils.common import initialize_integration_env_vars
    initialize_integration_env_vars()
    
    import threading
    import time
    from datetime import datetime, timedelta
    
    last_refresh = datetime.now()
    refresh_interval = 10  # segundos
    
    def show_menu_with_refresh_info():
        """Muestra el menú con información de última actualización"""
        _header_with_context("AutoTrainV2 - Main Menu", ["AutoTrainV2"])

    while True:
        current_time = datetime.now()
        
        # Verificar si han pasado 10 segundos desde la última actualización
        if (current_time - last_refresh).total_seconds() >= refresh_interval:
            last_refresh = current_time
            # Limpiar caché para obtener datos frescos
            _clear_caches()
        
        show_menu_with_refresh_info()
        
        choice = q.select(
            "",
            choices=[
                # Data Management
                q.Separator("╔════════════════════╗"),
                q.Separator("║  Data Management   ║"),
                q.Separator("╚════════════════════╝"),
                q.Choice(" » 1. Datasets", "datasets"),
                q.Choice("   2. Presets", "presets"),
                q.Separator("---------------"),
                
                # Train & Jobs
                q.Separator("╔══════════════════╗"),
                q.Separator("║  Train & Jobs    ║"),
                q.Separator("╚══════════════════╝"),
                q.Choice("   3. Training", "training"),
                q.Choice("   4. QuickTrain (1-Step)", "quicktrain"),
                q.Choice("   5. Jobs", "jobs"),
                q.Choice("   6. Logs", "logs"),
                q.Separator("---------------"),
                
                # Analysis
                q.Separator("╔════════════════╗"),
                q.Separator("║  Analysis      ║"),
                q.Separator("╚════════════════╝"),
                q.Choice("   7. Experiments", "experiments"),
                q.Choice("   8. Model Organizer", "organizer"),
                q.Separator("---------------"),
                
                # Tools
                q.Separator("╔════════════╗"),
                q.Separator("║  Tools     ║"),
                q.Separator("╚════════════╝"),
                q.Choice("   9. Web Interface", "web"),
                q.Choice("   0. Integrations", "integrations"),
                q.Separator("---------------"),
                
                # System
                q.Separator("╔═════════════╗"),
                q.Separator("║  System     ║"),
                q.Separator("╚═════════════╝"),
                q.Choice("   s. Status", "status"),
                q.Choice("   r. Refresh Now", "refresh"),
                q.Separator("---------------"),
                
                q.Choice("   q. Exit", "exit"),
            ],
            instruction="",
        ).ask()

        # Handle menu choices
        if choice in ['q', 'exit', None]:
            _header_with_context("Exiting AutoTrainV2", show_stats=False)
            console.print("🔄 Shutting down automatic training system...")
            break
        elif choice == 'refresh':
            # Actualizar inmediatamente
            last_refresh = datetime.now()
            _clear_caches()
            continue
        elif choice == 'datasets':
            _datasets_menu()
        elif choice == 'presets':
            _presets_menu()
        elif choice == 'training':
            _training_menu()
        elif choice == 'quicktrain':
            _quicktrain_menu()
        elif choice == 'jobs':
            _jobs_menu()
        elif choice == 'logs':
            _logs_menu()
        elif choice == 'experiments':
            _experiments_menu()
        elif choice == 'organizer':
            _organizer_menu()
        elif choice == 'web':
            _web_menu()
        elif choice == 'integrations':
            _integrations_menu()
        elif choice == 'status':
            _header_with_context("System Status", ["AutoTrainV2", "System"])
            _show_system_status()
            _pause()

# live view function (declared before executing main)

def _jobs_live_view():
    try:
        while True:
            jobs = _JOB_MANAGER.list_jobs()
            console.clear()
            grid = Table("ID", "Dataset", "Profile", "Status", "Progress")
            for j in jobs:
                prog = j.progress_str or (f"{j.percent}%" if j.percent else "")
                grid.add_row(j.id, j.dataset, j.profile, j.status.value, prog)
            console.print(grid)
            console.print("Press Ctrl-C to exit…")
            import time

            time.sleep(2)
    except KeyboardInterrupt:
        pass

# ---------------------------------------------------------------------------
# Integrations menu
# ---------------------------------------------------------------------------

def _integrations_menu():
    while True:
        _header_with_context("Integrations", ["AutoTrainV2", "Integrations"])
        
        # Show current status table
        from rich.table import Table as _Tbl
        tbl = _Tbl(title="Integrations status", box=None)
        tbl.add_column("Name", style="cyan")
        tbl.add_column("Enabled", style="green")
        for name in sorted(INTEGRATION_FLAGS):
            enabled = "[green]YES[/green]" if _integration_status(name) else "[red]NO[/red]"
            tbl.add_row(name, enabled)
        console.print(tbl)
        console.print()

        action = q.select(
            "Choose integration option:",
            choices=[
                # 🔧 TOGGLE
                q.Separator("╔════════════════════╗"),
                q.Separator("║  🔧 TOGGLE STATUS  ║"),
                q.Separator("╚════════════════════╝"),
                q.Choice(" » 0. Enable/Disable integration", "toggle"),
                # 🔗 CONFIGURE
                q.Separator("╔═══════════════════════╗"),
                q.Separator("║  🔗 CONFIGURE DETAILS ║"),
                q.Separator("╚═══════════════════════╝"),
                q.Choice("   1. Google Sheets integration", "gsheets"),
                q.Choice("   2. Hugging Face Hub", "huggingface"),
                q.Choice("   3. Remote output storage", "remote"),
                q.Separator("─────────────────────────"),
                q.Choice("   9. Back to main menu", "back"),
            ],
        ).ask()

        if action in {None, "back"}:
            return

        cfg = _load_int_cfg()

        # ---------- Toggle enable/disable ----------
        if action == "toggle":
            # Select integration to toggle
            name_sel = q.select("Select integration:", choices=sorted(INTEGRATION_FLAGS.keys()) + ["Cancel"]).ask()
            if not name_sel or name_sel == "Cancel":
                continue
            var = INTEGRATION_FLAGS[name_sel]
            current = cfg.get(var, os.getenv(var, "0"))
            new_val = "0" if str(current) == "1" else "1"
            cfg[var] = new_val
            _save_int_cfg(cfg)
            state_str = "enabled" if new_val == "1" else "disabled"
            console.print(f"[green]Integration '{name_sel}' {state_str}.[/green]")
            _pause()
            continue

        if action == "gsheets":
            current_cred = cfg.get("AUTO_GSHEET_CRED", "") or ""
            current_id = cfg.get("AUTO_GSHEET_ID", "") or ""
            current_tab = cfg.get("AUTO_GSHEET_TAB", "") or ""

            cred_path = q.path("Service-account JSON path:", default=current_cred or None).ask()
            if not cred_path:
                continue
            sheet_id = q.text("Spreadsheet ID:", default=current_id).ask()
            if not sheet_id:
                continue
            tab_name = q.text("Worksheet name (blank = first sheet):", default=current_tab).ask()

            # --- Variable selection ---
            from autotrain_sdk.integrations import GSHEET_HEADER as _GSHEET_HEADER
            current_keys_csv = cfg.get("AUTO_GSHEET_KEYS", "") or ""
            current_keys = [k.strip() for k in current_keys_csv.split(",") if k.strip()]
            choices_vars = [q.Choice(k, checked=(k in current_keys) or (not current_keys)) for k in _GSHEET_HEADER]
            selected_vars = q.checkbox("Select variables to include in Google Sheets:", choices=choices_vars).ask()
            if selected_vars is None:
                selected_vars = _GSHEET_HEADER  # fallback to all

            # Reorder prompt
            default_order = ",".join(selected_vars)
            new_order_csv = q.text("Specify order (comma separated) or leave blank:", default=default_order).ask()
            if new_order_csv:
                order_list_raw = [k.strip() for k in new_order_csv.split(",") if k.strip()]
                final_keys = [k for k in order_list_raw if k in selected_vars]
                # append any selected but missing to preserve
                for k in selected_vars:
                    if k not in final_keys:
                        final_keys.append(k)
            else:
                final_keys = selected_vars

            cfg.update({
                "AUTO_GSHEET_CRED": cred_path,
                "AUTO_GSHEET_ID": sheet_id.strip(),
                "AUTO_GSHEET_TAB": (tab_name.strip() if tab_name else None),
                "AUTO_GSHEET_KEYS": ",".join(final_keys),
                "AUTO_GSHEET_ENABLE": cfg.get("AUTO_GSHEET_ENABLE", "1"),
            })

        elif action == "huggingface":
            token = q.text("HF token:", default=cfg.get("AUTO_HF_TOKEN", ""), qmark="🐼").ask()
            repo = q.text("Repo id (username/model):", default=cfg.get("AUTO_HF_REPO", "")).ask()
            enable = q.confirm("Enable upload?", default=cfg.get("AUTO_HF_ENABLE", "0") == "1").ask()
            private = q.confirm("Private repo?", default=cfg.get("AUTO_HF_PRIVATE", "0") == "1").ask()
            cfg.update({
                "AUTO_HF_TOKEN": token or None,
                "AUTO_HF_REPO": repo or None,
                "AUTO_HF_ENABLE": "1" if enable else "0",
                "AUTO_HF_PRIVATE": "1" if private else "0",
            })



        elif action == "remote":
            base = q.path("Remote base path:", default=cfg.get("AUTO_REMOTE_BASE", "") or None).ask()
            direct = q.confirm("Write directly to remote path?", default=cfg.get("AUTO_REMOTE_DIRECT", "0") == "1").ask()
            cfg.update({
                "AUTO_REMOTE_BASE": base or None,
                "AUTO_REMOTE_DIRECT": "1" if direct else "0",
                "AUTO_REMOTE_ENABLE": cfg.get("AUTO_REMOTE_ENABLE", "1"),
            })

        _save_int_cfg(cfg)
        console.print("[green]Settings saved[/green]")
        _pause()

# ---------------------------------------------------------------------------
# Experiments (A/B testing) CLI
# ---------------------------------------------------------------------------

def _experiments_menu():
    """Menu to create and manage A/B experiments over hyper-parameters."""

    from autotrain_sdk.paths import BATCH_CONFIG_DIR
    from autotrain_sdk.experiments import create_experiment, EXP_DIR, Experiment
    from autotrain_sdk.job_manager import JOB_MANAGER, JobStatus
    import json

    def _list_dataset_names():
        from autotrain_sdk.paths import INPUT_DIR as _INP
        from autotrain_sdk.dataset_sources import list_external_datasets as _list_ext
        names = {p.name for p in _INP.glob("*") if p.is_dir()}
        names.update(_list_ext().keys())
        return sorted(names)

    def _parse_variation(val_str: str, current_val):
        val_str = (val_str or "").strip()
        if not val_str:
            return [current_val]
        # range a:b:c
        if ":" in val_str:
            parts = val_str.split(":")
            if len(parts) in (2, 3) and all(p.strip().replace("-", "").replace(".", "", 1).isdigit() for p in parts):
                start = float(parts[0]); stop = float(parts[1]); step = float(parts[2]) if len(parts)==3 else 1
                seq = []
                v = start
                while (step > 0 and v <= stop) or (step < 0 and v >= stop):
                    seq.append(type(current_val)(v))
                    v += step
                return seq
        # comma list
        items = [x.strip() for x in val_str.split(",") if x.strip()]
        if isinstance(current_val, (int, float)):
            def cast(x):
                try:
                    return type(current_val)(float(x))
                except Exception:
                    return current_val
            return [cast(x) for x in items]
        return items

    def _generate_variants(base_cfg: dict, overrides_input: dict[str, str]):
        keys = []
        values_list = []
        for arg, var_str in overrides_input.items():
            cur_val = base_cfg.get(arg)
            if cur_val is None:
                continue
            options = _parse_variation(var_str, cur_val)
            if len(options) == 1 and options[0] == cur_val:
                continue
            keys.append(arg)
            values_list.append(options)
        import itertools
        variants = []
        for combo in itertools.product(*values_list):
            ov = dict(zip(keys, combo))
            if any(base_cfg.get(k) != v for k, v in ov.items()):
                variants.append(ov)
        if not variants:
            variants.append({})
        return variants

    while True:
        _header_with_context("A/B Experiments", ["AutoTrainV2", "Experiments"])
        
        # Show current experiments status
        exp_count = len(list(EXP_DIR.glob("*.json")))
        console.print(f"[dim]🔬 {exp_count} experiment(s) configured[/dim]")
        console.print()
        
        choice = q.select(
            "Choose action:",
            choices=[
                # 🚀 EXPERIMENT MANAGEMENT
                q.Separator("╔═══════════════════════════╗"),
                q.Separator("║  🚀 EXPERIMENT MANAGEMENT ║"),
                q.Separator("╚═══════════════════════════╝"),
                q.Choice(" » 1. Launch new experiment", "launch"),
                q.Choice("   2. List experiments", "list"),
                q.Choice("   3. Compare variants", "compare"),
                q.Separator("───────────────────────────"),
                
                # 🗑️ CLEANUP
                q.Separator("╔═══════════════╗"),
                q.Separator("║  🗑️ CLEANUP  ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   4. Clear experiments", "clear"),
                q.Separator("─────────────────"),
                
                q.Choice("   0. Back to main menu", "back"),
            ],
        ).ask()

        if choice in {None, "back"}:
            return

        if choice == "launch":
            datasets = _list_dataset_names()
            if not datasets:
                console.print("[yellow]No datasets available[/yellow]")
                _pause()
                continue
            dataset = q.select("Dataset:", choices=datasets).ask()
            if not dataset:
                continue
            profile = q.select("Base profile:", choices=["Flux", "FluxLORA", "Nude"]).ask()
            if not profile:
                continue

            path_base = BATCH_CONFIG_DIR / profile / f"{dataset}.toml"
            try:
                from autotrain_sdk.configurator import load_config
                cfg_base = load_config(path_base)
            except Exception as e:
                _error(f"Base TOML not found: {e}")
                continue

            # Ask overrides (loop)
            overrides: dict[str, str] = {}
            while True:
                key = q.text("Parameter to vary (blank to finish):").ask()
                if not key:
                    break
                cur = cfg_base.get(key, "<undefined>")
                val = q.text(f"Variations for '{key}' (current {cur}):").ask()
                overrides[key] = val

            variants = _generate_variants(cfg_base, overrides)
            variants_payload = [{"profile": profile, "overrides": ov} for ov in variants]

            exp = create_experiment(dataset, variants_payload)
            console.print(f"[green]Experiment {exp.id} launched with {len(variants)} variants[/green]")
            _pause()

        elif choice == "list":
            rows = []
            for p in EXP_DIR.glob("*.json"):
                try:
                    data = json.loads(p.read_text())
                    exp_id = data["id"]
                    runs_total = len(data["runs"])
                    jobs = JOB_MANAGER.list_jobs()
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
                except Exception:
                    pass

            if not rows:
                console.print("[yellow]No experiments found[/yellow]")
            else:
                tbl = Table("ID", "Dataset", "#Runs", "Status")
                for r in rows:
                    tbl.add_row(*map(str, r))
                console.print(tbl)
            _pause()

        elif choice == "compare":
            exps = [p.stem for p in EXP_DIR.glob("*.json")]
            if not exps:
                console.print("[yellow]No experiments available[/yellow]")
                _pause()
                continue
            exp_id = q.select("Select experiment:", choices=exps).ask()
            if not exp_id:
                continue
            try:
                exp = Experiment.load(exp_id)
            except Exception as e:
                _error(str(e))
                continue

            # Load run registry for job mapping
            from autotrain_sdk.run_registry import _load as _rr_load
            recs_exp = [rec for rec in _rr_load().values() if rec.get("experiment_id") == exp_id]

            def _job_for_idx(i: int) -> str:
                tag = f"tmp_{exp_id}_{i}.toml"
                for rec in recs_exp:
                    if tag in str(rec.get("toml_path", "")):
                        return rec.get("job_id", "-")
                return "-"

            tbl = Table("JobID", "Profile", "Overrides")
            for idx, run in enumerate(exp.runs):
                prof = run.get("profile", "Flux")
                ov = run.get("overrides", {})
                job_code = _job_for_idx(idx)
                ov_str = ", ".join(f"{k}={v}" for k, v in ov.items()) or "-"
                tbl.add_row(job_code, prof, ov_str)
            console.print(tbl)
            _pause()

        elif choice == "clear":
            if not q.confirm("This will delete all experiment JSON files. Continue?", default=False).ask():
                continue
            cnt = 0
            for p in EXP_DIR.glob("*.json"):
                try:
                    p.unlink()
                    cnt += 1
                except Exception:
                    pass
            console.print(f"[red]Deleted {cnt} experiment file(s)[/red]")
            _pause()

# ---------------------------------------------------------------------------
# Model Organizer CLI (view and filter run registry)
# ---------------------------------------------------------------------------

def _organizer_menu():
    """CLI version of the Gradio *Model Organizer* tab."""

    from autotrain_sdk.run_registry import _load as _rr_load, _save as _rr_save

    def _filter_records(query: str, status: str):
        q = (query or "").strip().lower()
        recs = _rr_load().values()
        rows = []
        for rec in recs:
            if status != "all" and rec.get("status") != status:
                continue
            if q and q not in rec.get("dataset", "").lower() and q not in rec.get("profile", "").lower() and q not in rec.get("job_id", "").lower():
                continue
            rows.append(rec)
        return rows

    while True:
        _header_with_context("Model Organizer", ["AutoTrainV2", "Organizer"])
        
        # Show run registry stats
        all_records = list(_rr_load().values())
        status_counts = {}
        for rec in all_records:
            status = rec.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if all_records:
            status_items = []
            for status, count in status_counts.items():
                emoji = {
                    "pending": "⏳",
                    "running": "🔄",
                    "done": "✅",
                    "failed": "❌",
                    "canceled": "⏹️"
                }.get(status, "📋")
                status_items.append(f"{emoji} {status}: {count}")
            
            console.print(f"[dim]📊 {len(all_records)} total runs | {' | '.join(status_items)}[/dim]")
        else:
            console.print("[dim]📊 No run records found[/dim]")
        console.print()

        action = q.select(
            "Choose action:",
            choices=[
                # 👁️ VIEWING
                q.Separator("╔═══════════════╗"),
                q.Separator("║  👁️ VIEWING   ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice(" » 1. List runs", "list"),
                q.Choice("   2. Show run details", "details"),
                q.Choice("   3. Search / Filter", "search"),
                q.Separator("─────────────────"),
                
                # 🗑️ CLEANUP
                q.Separator("╔═══════════════╗"),
                q.Separator("║  🗑️ CLEANUP  ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   4. Clear records", "clear"),
                q.Separator("─────────────────"),
                
                q.Separator(),
                q.Choice("0. Back to main menu", "back"),
            ],
        ).ask()

        if action in {None, "back"}:
            return

        if action == "list":
            rows = _filter_records("", "all")
            if not rows:
                console.print("[yellow]No records found[/yellow]")
            else:
                tbl = Table("JobID", "Dataset", "Profile", "Status", "Epochs", "Batch", "LR", "FID", "CLIP")
                for rec in rows:
                    tbl.add_row(
                        rec.get("job_id", "-"),
                        rec.get("dataset", "-"),
                        rec.get("profile", "-"),
                        rec.get("status", "-"),
                        str(rec.get("epochs", "-")),
                        str(rec.get("batch_size", "-")),
                        str(rec.get("learning_rate", "-")),
                        str(rec.get("fid", "-")),
                        str(rec.get("clip", "-")),
                    )
                console.print(tbl)
            _pause()

        elif action == "search":
            query = q.text("Search term (dataset, profile, id):").ask()
            status = q.select("Status:", choices=["all", "pending", "running", "done", "failed", "canceled"], default="all").ask()
            rows = _filter_records(query or "", status)
            if not rows:
                console.print("[yellow]No matching records\n[/yellow]")
            else:
                tbl = Table("JobID", "Dataset", "Profile", "Status")
                for rec in rows:
                    tbl.add_row(rec.get("job_id", "-"), rec.get("dataset", "-"), rec.get("profile", "-"), rec.get("status", "-"))
                console.print(tbl)
            _pause()

        elif action == "details":
            recs = list(_rr_load().values())
            ids = [r.get("job_id") for r in recs]
            if not ids:
                console.print("[yellow]No records available[/yellow]")
                _pause()
                continue
            jid = q.select("Select JobID:", choices=ids).ask()
            if not jid:
                continue
            rec = next((r for r in recs if r.get("job_id") == jid), None)
            if not rec:
                _error("Job not found in registry")
                continue
            # show details
            md = Table.grid()
            for k in ["dataset", "profile", "status", "epochs", "batch_size", "learning_rate", "fid", "clip", "queued_time", "start_time", "end_time"]:
                if k in rec and rec[k] is not None:
                    md.add_row(f"[bold]{k}[/bold]", str(rec[k]))
            console.print(md)
            # show sample images paths (last 6 png)
            from pathlib import Path
            run_path = Path(rec.get("run_dir", ""))
            imgs = sorted(run_path.rglob("*.png"))[-6:]
            if imgs:
                console.print("\nSample images:")
                for p in imgs:
                    console.print(f" • {p}")
            _pause()

        elif action == "clear":
            if not q.confirm("Delete ALL run records?", default=False).ask():
                continue
            _rr_save({})
            console.print("[red]Registry cleared[/red]")
            _pause()

# ---------------------------------------------------------------------------
# Enhanced Training Functions (from CLI)
# ---------------------------------------------------------------------------

def _format_job_status_menu(status: JobStatus) -> str:
    """Format job status with colored emoji for menu."""
    status_map = {
        JobStatus.RUNNING: "🟢 RUNNING",
        JobStatus.PENDING: "⏸️ PENDING", 
        JobStatus.DONE: "✅ DONE",
        JobStatus.FAILED: "❌ FAILED",
        JobStatus.CANCELED: "🚫 CANCELED"
    }
    return status_map.get(status, str(status))

def _create_dataset_info_table_menu(dataset_name: str, profile: str) -> Table:
    """Create a rich table with dataset information for menu."""
    try:
        from .gradio_app import _get_dataset_info
        dataset_info = _get_dataset_info(dataset_name)
    except (ImportError, AttributeError):
        dataset_info = _get_dataset_info_fallback(dataset_name)
    
    table = Table(title=f"Dataset Information: {dataset_name}", show_header=False, box=None)
    table.add_column("Property", style="bold blue")
    table.add_column("Value", style="cyan")
    
    if not dataset_info.get("exists", False):
        table.add_row("❌ Error", dataset_info.get("error", "Dataset not found"))
        return table
    
    table.add_row("🖼️ Images", str(dataset_info.get("num_images", 0)))
    table.add_row("📝 Captions", str(dataset_info.get("num_captions", 0)))
    table.add_row("📐 Avg Resolution", dataset_info.get("avg_resolution", "N/A"))
    table.add_row("💾 Total Size", dataset_info.get("total_size", "N/A"))
    table.add_row("📅 Last Modified", dataset_info.get("last_modified", "N/A"))
    table.add_row("📁 Path", f"`{dataset_info.get('path', 'N/A')}`")
    
    return table

def _create_training_estimation_table_menu(dataset_name: str, profile: str) -> Table:
    """Create a rich table with training time estimation for menu."""
    try:
        from .gradio_app import _estimate_training_time
        estimation = _estimate_training_time(dataset_name, profile)
    except (ImportError, AttributeError):
        estimation = _estimate_training_time_fallback(dataset_name, profile)
    
    table = Table(title=f"Training Estimation: {profile}", show_header=False, box=None)
    table.add_column("Property", style="bold green")
    table.add_column("Value", style="yellow")
    
    if "error" in estimation:
        table.add_row("❌ Error", estimation["error"])
        return table
    
    table.add_row("⏰ Estimated Time", estimation.get("estimated_time", "N/A"))
    table.add_row("🔄 Total Steps", f"{estimation.get('total_steps', 0):,}")
    table.add_row("", "")  # Separator
    table.add_row("📊 Parameters", "")
    
    params = estimation.get("parameters", {})
    table.add_row("  • Epochs", str(params.get("epochs", "N/A")))
    table.add_row("  • Batch Size", str(params.get("batch_size", "N/A")))
    table.add_row("  • Repeats", str(params.get("repeats", "N/A")))
    
    return table

def _create_queue_table_menu() -> Table:
    """Create a rich table showing the training queue for menu."""
    jobs = _JOB_MANAGER.list_jobs()
    
    table = Table(title="Training Queue", box=None)
    table.add_column("Job ID", style="cyan")
    table.add_column("Dataset", style="blue") 
    table.add_column("Profile", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Progress", style="green")
    
    for job in jobs:
        progress_text = "Waiting in queue"
        if job.status == JobStatus.RUNNING and job.progress_str:
            progress_text = f"{job.percent:.1f}% ({job.current_step}/{job.total_steps}) ETA {job.eta}"
        elif job.status == JobStatus.DONE:
            progress_text = "Completed"
        elif job.status == JobStatus.FAILED:
            progress_text = "Failed"
        elif job.status == JobStatus.CANCELED:
            progress_text = "Canceled"
        
        table.add_row(
            job.id,
            job.dataset,
            job.profile,
            _format_job_status_menu(job.status),
            progress_text
        )
    
    if not jobs:
        table.add_row("", "", "", "", "No jobs in queue")
    
    return table

def _show_enhanced_metrics_menu(job: Job):
    """Show enhanced metrics for a job in menu format."""
    try:
        metrics_data = _JOB_MANAGER.get_metrics_data(job.id)
    except AttributeError:
        console.print("[yellow]⏳ Metrics not available (get_metrics_data not implemented)[/yellow]")
        return
    
    if not metrics_data:
        console.print("[yellow]⏳ Waiting for training metrics...[/yellow]")
        return
    
    # Format metrics using fallback function
    try:
        from .gradio_app import _format_enhanced_metrics
        metrics_text = _format_enhanced_metrics(metrics_data)
    except (ImportError, AttributeError):
        metrics_text = _format_enhanced_metrics_fallback(metrics_data)
    
    # Display in a panel
    console.print(Panel(metrics_text, title="📊 Enhanced Metrics", border_style="blue"))

def _show_training_logs_menu(job: Job, lines: int = 20):
    """Show recent training logs for a job in menu format."""
    try:
        log, _ = _JOB_MANAGER.get_live_output(job.id)
    except AttributeError:
        console.print("[yellow]⏳ Live output not available (get_live_output not implemented)[/yellow]")
        return
    
    if not log:
        console.print("[yellow]⏳ Waiting for training log...[/yellow]")
        return
    
    # Show last N lines
    log_lines = log.split('\n')[-lines:]
    log_content = '\n'.join(log_lines)
    
    console.print(Panel(log_content, title="📝 Training Log", border_style="red"))

# ---------------------------------------------------------------------------
# Enhanced Training Menu Functions
# ---------------------------------------------------------------------------

def _training_info_menu():
    """Enhanced dataset information and training estimation menu."""
    _header_with_context("Training Information", ["AutoTrainV2", "Training", "Dataset Info"])
    
    # Get available datasets
    try:
        datasets_info = _get_datasets_cached()
        available = datasets_info['all']
    except Exception:
        from autotrain_sdk.paths import INPUT_DIR
        available = [p.name for p in INPUT_DIR.glob("*") if p.is_dir()]
    
    if not available:
        console.print("[yellow]No datasets found[/yellow]")
        _pause()
        return
    
    # Select dataset
    dataset_name = q.select("Select dataset for information:", choices=available + ["Back"]).ask()
    if not dataset_name or dataset_name == "Back":
        return
    
    # Select profile
    profile = q.select("Select training profile:", 
                      choices=["Flux", "FluxLORA", "Nude"], 
                      default="FluxLORA").ask()
    if not profile:
        return
    
    console.clear()
    _header_with_context(f"Dataset Analysis: {dataset_name}", ["AutoTrainV2", "Training", "Dataset Info"])
    
    # Show dataset information
    info_table = _create_dataset_info_table_menu(dataset_name, profile)
    console.print(info_table)
    console.print()
    
    # Show training estimation
    estimation_table = _create_training_estimation_table_menu(dataset_name, profile)
    console.print(estimation_table)
    console.print()
    
    _pause()

def _training_queue_menu():
    """Enhanced training queue management menu."""
    while True:
        _header_with_context("Training Queue Management", ["AutoTrainV2", "Training", "Queue"])
        
        # Show queue table
        queue_table = _create_queue_table_menu()
        console.print(queue_table)
        console.print()
        
        action = q.select(
            "Choose queue action:",
            choices=[
                q.Choice("1. Refresh view", "refresh"),
                q.Choice("2. Cancel job", "cancel"),
                q.Choice("3. Job details", "details"),
                q.Choice("4. Clean completed", "clean"),
                q.Separator(),
                q.Choice("0. Back", "back"),
            ],
        ).ask()
        
        if action in {None, "back"}:
            return
        
        if action == "refresh":
            continue  # Loop will refresh automatically
        
        elif action == "cancel":
            jobs = [j for j in _JOB_MANAGER.list_jobs() 
                   if j.status in {JobStatus.PENDING, JobStatus.RUNNING}]
            if not jobs:
                console.print("[yellow]No cancelable jobs[/yellow]")
                _pause()
                continue
            
            job_choices = [f"{j.id} ({j.dataset}, {j.profile})" for j in jobs]
            selected = q.select("Select job to cancel:", choices=job_choices + ["Cancel"]).ask()
            if not selected or selected == "Cancel":
                continue
            
            job_id = selected.split(" ")[0]
            if q.confirm(f"Cancel job {job_id}?", default=False).ask():
                _JOB_MANAGER.cancel(job_id)
                console.print(f"[green]✅ Job {job_id} canceled[/green]")
                _pause()
        
        elif action == "details":
            jobs = _JOB_MANAGER.list_jobs()
            if not jobs:
                console.print("[yellow]No jobs in queue[/yellow]")
                _pause()
                continue
            
            job_choices = [f"{j.id} ({j.dataset}, {j.profile}, {j.status.value})" for j in jobs]
            selected = q.select("Select job for details:", choices=job_choices + ["Cancel"]).ask()
            if not selected or selected == "Cancel":
                continue
            
            job_id = selected.split(" ")[0]
            job = next((j for j in jobs if j.id == job_id), None)
            if job:
                _show_job_details_menu(job)
        
        elif action == "clean":
            jobs = _JOB_MANAGER.list_jobs()
            completed_jobs = [j for j in jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
            
            if not completed_jobs:
                console.print("[yellow]No completed jobs to clean[/yellow]")
                _pause()
                continue
            
            console.print(f"[yellow]Found {len(completed_jobs)} completed job(s)[/yellow]")
            if q.confirm(f"Remove {len(completed_jobs)} completed job(s) from queue?", default=True).ask():
                removed = 0
                for job in completed_jobs:
                    try:
                        if _JOB_MANAGER.remove_job(job.id):
                            removed += 1
                    except AttributeError:
                        # remove_job not implemented, use alternative method
                        console.print(f"[yellow]⚠️  remove_job not implemented - completed jobs will remain in queue[/yellow]")
                        break
                console.print(f"[green]✅ Cleaned {removed} job(s) from queue[/green]")
                _pause()

def _show_job_details_menu(job: Job):
    """Show detailed information about a specific job."""
    _header_with_context(f"Job Details: {job.id}", ["AutoTrainV2", "Training", "Job Details"])
    
    # Basic job information
    info_table = Table("Property", "Value", show_header=False, box=None)
    info_table.add_row("🆔 Job ID", job.id)
    info_table.add_row("📂 Dataset", job.dataset)
    info_table.add_row("⚙️ Profile", job.profile)
    info_table.add_row("📊 Status", _format_job_status_menu(job.status))
    
    if job.progress_str:
        info_table.add_row("📈 Progress", job.progress_str)
    if job.avg_loss is not None:
        info_table.add_row("📉 Loss", f"{job.avg_loss:.4f}")
    if hasattr(job, 'gpu_ids') and job.gpu_ids:
        info_table.add_row("🎮 GPU IDs", job.gpu_ids)
    
    console.print(info_table)
    console.print()
    
    # Enhanced metrics if running
    if job.status == JobStatus.RUNNING:
        _show_enhanced_metrics_menu(job)
        console.print()
    
    # Recent logs
    console.print("[bold]Recent logs:[/bold]")
    _show_training_logs_menu(job, lines=6)
    console.print()
    
    _pause()



def _get_terminal_size():
    """Get terminal size with fallback for different environments."""
    try:
        # Try Rich console first
        width = console.size.width
        height = console.size.height
        
        # Validate reasonable sizes
        if width < 40 or height < 10:
            raise ValueError("Terminal too small")
        
        return width, height
    except:
        # Fallback to os.get_terminal_size()
        try:
            import os
            size = os.get_terminal_size()
            return size.columns, size.lines
        except:
            # Final fallback
            return 80, 24

def _create_simple_job_display(job: Job):
    """Create enhanced job display with better aesthetics using Rich objects."""
    from datetime import datetime
    from rich.text import Text
    from rich.console import Group
    
    # Get terminal dimensions
    terminal_width, terminal_height = _get_terminal_size()
    terminal_width = max(50, terminal_width)
    
    # Get output path early to adjust border width if needed
    output_path = "N/A"
    if hasattr(job, 'output_dir') and job.output_dir:
        output_path = str(job.output_dir)
    elif hasattr(job, 'run_dir') and job.run_dir:
        output_path = str(job.run_dir)
    
    # Calculate minimum border width needed for output path
    min_width_for_output = len(f"💾 Output: {output_path}") + 6
    
    # Create border width based on terminal and content
    border_width = max(60, min(terminal_width - 4, min_width_for_output))
    
    # Create simple content string
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Create lines as Rich Text objects
    lines = []
    
    # Helper function to remove Rich markup for length calculation
    def _strip_markup(text):
        """Remove Rich markup codes to get visual length."""
        import re
        # Remove [style] and [/style] patterns
        cleaned = re.sub(r'\[/?[^\]]*\]', '', str(text))
        return cleaned
    
    # Helper function to create padded lines
    def _create_padded_line(emoji_label, value, label_style="white", value_style="white"):
        # Create the line content
        line_text = Text("│ ", style="white")
        line_text.append(emoji_label, style=label_style)
        line_text.append(" ", style="white")
        line_text.append(value, style=value_style)
        
        # Calculate padding using visual length (without markup)
        visual_content = f"{_strip_markup(emoji_label)} {_strip_markup(value)}"
        content_length = len(visual_content)
        padding = max(0, border_width - content_length - 5)
        line_text.append(" " * padding, style="white")
        line_text.append("│", style="white")
        
        return line_text
    
    # Header with border
    lines.append(Text("╭" + "─" * (border_width - 2) + "╮", style="white"))
    
    # Helper function for header lines with proper padding
    def _create_header_line(content, style="white"):
        line_text = Text("│ ", style="white")
        line_text.append(content, style=style)
        # Calculate padding using visual length
        visual_length = len(_strip_markup(content))
        padding = max(0, border_width - visual_length - 3)
        line_text.append(" " * padding, style="white")
        line_text.append("│", style="white")
        return line_text
    
    # Header lines
    job_display_id = job.id[:12] + ('...' if len(job.id) > 12 else '')
    lines.append(_create_header_line(f"🔍 Live Monitor - Job: {job_display_id}", "bold cyan"))
    lines.append(_create_header_line(f"⏰ {current_time} | Press Ctrl+C to exit", "dim"))
    
    lines.append(Text("├" + "─" * (border_width - 2) + "┤", style="white"))
    
    # Job details section
    if terminal_width < 60:
        lines.append(_create_padded_line("🆔 ID:", job.id[:12] + "..."))
        lines.append(_create_padded_line("📂 Dataset:", job.dataset[:15] + ('...' if len(job.dataset) > 15 else '')))
        lines.append(_create_padded_line("⚙️ Profile:", job.profile[:10] + ('...' if len(job.profile) > 10 else '')))
    else:
        lines.append(_create_padded_line("🆔 Job ID:", job.id))
        lines.append(_create_padded_line("📂 Dataset:", job.dataset))
        lines.append(_create_padded_line("⚙️ Profile:", job.profile))
    
    # Status and progress section
    lines.append(Text("├" + "─" * (border_width - 2) + "┤", style="white"))
    
    # Status line (special handling for formatted status)
    status_formatted = _format_job_status_menu(job.status)
    
    # For status line, we need special handling since status_formatted might contain markup
    status_line = Text("│ ", style="white")
    status_line.append("📊 Status:", style="white")
    status_line.append(" ", style="white")
    status_line.append(status_formatted, style="white")  # Status already has formatting
    
    # Calculate padding using visual length
    visual_status = f"📊 Status: {_strip_markup(status_formatted)}"
    status_padding = max(0, border_width - len(visual_status) - 3)
    status_line.append(" " * status_padding, style="white")
    status_line.append("│", style="white")
    lines.append(status_line)
    
    # Helper function to read max_train_epochs from preset
    def _get_max_epochs_from_preset():
        try:
            # Try multiple ways to get the preset path
            preset_path = None
            
            if hasattr(job, 'preset_path') and job.preset_path:
                preset_path = job.preset_path
            elif hasattr(job, 'profile') and hasattr(job, 'dataset'):
                # Construct path from profile and dataset
                from autotrain_sdk.paths import BATCH_CONFIG_DIR
                potential_path = BATCH_CONFIG_DIR / job.profile / f"{job.dataset}.toml"
                if potential_path.exists():
                    preset_path = potential_path
            
            if preset_path:
                preset_data = toml.load(preset_path)
                # Try different possible keys
                for key in ['max_train_epochs', 'num_train_epochs', 'epochs', 'max_epochs']:
                    if key in preset_data:
                        return preset_data[key]
        except Exception as e:
            # Debug: could log the error for troubleshooting
            pass
        return 'N/A'
    
    # Always show progress (even if no data yet)
    if job.progress_str:
        # Show detailed progress with steps if available
        if hasattr(job, 'current_step') and hasattr(job, 'total_steps') and job.total_steps > 0:
            progress_text = f"{job.progress_str} ({job.current_step}/{job.total_steps})"
        else:
            progress_text = job.progress_str
        lines.append(_create_padded_line("📈 Progress:", progress_text, value_style="green"))
    else:
        lines.append(_create_padded_line("📈 Progress:", "Initiating training", value_style="yellow"))
    
    # Always show loss (even if no data yet)
    if job.avg_loss is not None:
        lines.append(_create_padded_line("📉 Loss:", f"{job.avg_loss:.4f}", value_style="yellow"))
    else:
        lines.append(_create_padded_line("📉 Loss:", "Initiating training", value_style="yellow"))
    
    # Show epochs info
    max_epochs = _get_max_epochs_from_preset()
    lines.append(_create_padded_line("🔄 Epochs:", f"{max_epochs}", value_style="cyan"))
    
    # Helper function to get GPU usage
    def _get_gpu_usage():
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if not gpus:
                return "No GPU detected"
            
            # If job has specific GPU IDs, show only those
            if hasattr(job, 'gpu_ids') and job.gpu_ids:
                gpu_ids = [int(id.strip()) for id in str(job.gpu_ids).split(',')]
                gpu_usage = []
                for gpu_id in gpu_ids:
                    if gpu_id < len(gpus):
                        gpu = gpus[gpu_id]
                        usage_percent = round(gpu.load * 100, 1)
                        gpu_usage.append(f"{usage_percent}%")
                return " | ".join(gpu_usage)
            else:
                # Show all GPUs with simplified format
                gpu_usage = []
                for gpu in gpus:
                    usage_percent = round(gpu.load * 100, 1)
                    gpu_usage.append(f"{usage_percent}%")
                return " | ".join(gpu_usage)
        except ImportError:
            # Fallback when GPUtil is not available
            return "Usage n/a"
        except Exception as e:
            return f"Error: {str(e)[:15]}..."
    
    # Show GPU usage instead of IDs
    gpu_usage = _get_gpu_usage()
    lines.append(_create_padded_line("🎮 GPU Usage:", gpu_usage, value_style="magenta"))
    
    # Output section (separate grid)
    lines.append(Text("├" + "─" * (border_width - 2) + "┤", style="white"))
    
    # Output path - already calculated above
    lines.append(_create_padded_line("💾 Output:", output_path, value_style="blue"))
    
    # Footer border
    lines.append(Text("╰" + "─" * (border_width - 2) + "╯", style="white"))
    
    # Return a Group object that Rich can render properly
    return Group(*lines)

def _training_monitor_menu():
    """Enhanced training monitor using Rich Live - monitors multiple jobs sequentially."""
    from rich.live import Live
    from rich.console import Group
    from rich.text import Text
    import time
    
    # Get current job or select one to start with
    try:
        current_job = _JOB_MANAGER.get_current_job()
    except AttributeError:
        # get_current_job not implemented, current_job will be None
        current_job = None
    
    if not current_job:
        # No current job, let user select one or auto-detect first running job
        running_jobs = [j for j in _JOB_MANAGER.list_jobs() if j.status == JobStatus.RUNNING]
        pending_jobs = [j for j in _JOB_MANAGER.list_jobs() if j.status == JobStatus.PENDING]
        
        if not running_jobs and not pending_jobs:
            console.print("[yellow]No running or pending jobs to monitor[/yellow]")
            _pause()
            return
        
        # Auto-select first running job, or first pending if no running
        if running_jobs:
            current_job = running_jobs[0]
            console.print(f"[green]Auto-selected running job: {current_job.id}[/green]")
        else:
            current_job = pending_jobs[0]
            console.print(f"[blue]Auto-selected pending job: {current_job.id}[/blue]")
    
    if not current_job:
        console.print("[yellow]No job available for monitoring[/yellow]")
        _pause()
        return
    
    console.print(f"[green]🔍 Starting Live Monitor for job {current_job.id}[/green]")
    console.print("[dim]Starting in 2 seconds... (Press Ctrl+C to stop monitoring all jobs)[/dim]")
    time.sleep(2)
    
    try:
        # Use full screen mode so previous frames are fully cleared even if the renderable height changes.
        # This avoids duplicated panels (e.g., system stats) when switching between jobs in the live monitor.
        with Live(console=console, refresh_per_second=1, auto_refresh=False, screen=True) as live:
            while True:
                # Monitor current job
                job_completed = False
                
                while True:
                    # Get updated job
                    updated_job = None
                    for j in _JOB_MANAGER.list_jobs():
                        if j.id == current_job.id:
                            updated_job = j
                            break
                    
                    if not updated_job:
                        live.update(Text("❌ Job no longer found in queue"))
                        live.refresh()
                        time.sleep(3)
                        job_completed = True
                        break
                    
                    # Check if job is still running
                    if updated_job.status not in [JobStatus.RUNNING, JobStatus.PENDING]:
                        final_message = f"✅ Job {updated_job.id} {updated_job.status.value}"
                        live.update(Text(final_message, style="green bold"))
                        live.refresh()
                        time.sleep(2)
                        job_completed = True
                        break
                    
                    # Create and display job info
                    try:
                        from rich.padding import Padding
                        stats_panel = _create_system_stats_panel()
                        # Add one line of top padding so the upper border isn't cut off in full-screen mode
                        content = Group(Padding(stats_panel, (1, 0, 0, 0)), _create_simple_job_display(updated_job))
                        # Build list of renderables for Live output
                        renderables = [
                            Padding(stats_panel, (1, 0, 0, 0)),
                            _create_simple_job_display(updated_job),
                        ]

                        upcoming_tbl = _create_upcoming_jobs_table()
                        if upcoming_tbl is not None:
                            renderables.append(upcoming_tbl)

                        content = Group(*renderables)
                        # Content is now a Group object that Rich can render directly
                        live.update(content)
                        live.refresh()
                    except Exception as e:
                        error_message = f"❌ Display error: {str(e)}\nLooking for next job..."
                        live.update(Text(error_message))
                        live.refresh()
                        time.sleep(3)
                        job_completed = True
                        break
                    
                    # Wait for next update
                    time.sleep(3)
                
                # Current job completed, look for next job
                if job_completed:
                    # Get next available job (running or pending)
                    all_jobs = _JOB_MANAGER.list_jobs()
                    next_jobs = [j for j in all_jobs if j.status in [JobStatus.RUNNING, JobStatus.PENDING] and j.id != current_job.id]
                    
                    if not next_jobs:
                        # No more jobs to monitor
                        live.update(Text("🎉 All jobs completed! Monitor will exit.", style="green bold"))
                        live.refresh()
                        time.sleep(3)
                        break
                    
                    # Show loading message for next job
                    next_job = next_jobs[0]  # Get the first available job
                    loading_message = f"🔄 Loading next job: {next_job.id} ({next_job.dataset}, {next_job.profile})"
                    live.update(Text(loading_message, style="blue bold"))
                    live.refresh()
                    time.sleep(3)
                    
                    # Switch to next job
                    current_job = next_job
                    continue
                
                # Should not reach here normally
                break
                
    except KeyboardInterrupt:
        console.print("\n[yellow]✅ Live Monitor stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Live Monitor error: {str(e)}[/red]")
        console.print("Monitor will exit...")
    
    _pause()

def _training_history_menu():
    """Enhanced training history viewer menu with cleanup options."""
    
    while True:
        _header_with_context("Training History", ["AutoTrainV2", "Jobs", "History"])
        
        jobs = _JOB_MANAGER.list_jobs()
        
        if not jobs:
            console.print("[yellow]No training history found[/yellow]")
            _pause()
            return
        
        # Show quick summary
        completed_jobs = [j for j in jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
        console.print(f"[dim]📊 Total: {len(jobs)} jobs | {len(completed_jobs)} completed[/dim]")
        console.print()
        
        action = q.select(
            "Choose action:",
            choices=[
                # 📜 VIEWING
                q.Separator("╔═══════════════╗"),
                q.Separator("║  📜 VIEWING   ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice(" » 1. View All Jobs", "view_all"),
                q.Choice("   2. View Completed Only", "view_done"),
                q.Choice("   3. View Failed Only", "view_failed"),
                q.Choice("   4. Filter by Dataset", "filter_dataset"),
                q.Choice("   5. Filter by Profile", "filter_profile"),
                q.Separator("─────────────────"),
                
                # 🧹 CLEANUP
                q.Separator("╔═══════════════╗"),
                q.Separator("║  🧹 CLEANUP   ║"),
                q.Separator("╚═══════════════╝"),
                q.Choice("   6. Clean Completed Jobs", "clean_completed"),
                q.Separator("─────────────────"),
                
                q.Choice("   0. Back", "back"),
            ],
        ).ask()
        
        if action == "back" or action is None:
            return
        elif action == "clean_completed":
            _clean_completed_jobs()
            continue
        
        # Handle viewing options
        filtered_jobs = jobs
        
        if action == "view_done":
            filtered_jobs = [j for j in jobs if j.status == JobStatus.DONE]
        elif action == "view_failed":
            filtered_jobs = [j for j in jobs if j.status == JobStatus.FAILED]
        elif action == "filter_dataset":
            datasets = sorted(set(j.dataset for j in jobs))
            selected_dataset = q.select("Select dataset:", choices=datasets + ["Cancel"]).ask()
            if not selected_dataset or selected_dataset == "Cancel":
                continue
            filtered_jobs = [j for j in jobs if j.dataset == selected_dataset]
        elif action == "filter_profile":
            profiles = sorted(set(j.profile for j in jobs))
            selected_profile = q.select("Select profile:", choices=profiles + ["Cancel"]).ask()
            if not selected_profile or selected_profile == "Cancel":
                continue
            filtered_jobs = [j for j in jobs if j.profile == selected_profile]
        
        if not filtered_jobs:
            console.print("[yellow]No jobs match the selected filter[/yellow]")
            _pause()
            continue
        
        # Sort by most recent first (by job ID which contains timestamp)
        filtered_jobs = sorted(filtered_jobs, key=lambda x: x.id, reverse=True)
        
        # Create history table
        console.clear()
        _header_with_context("Training History", ["AutoTrainV2", "Jobs", "History"])
        
        table = Table(title=f"Training History ({len(filtered_jobs)} jobs)")
        table.add_column("Job ID", style="cyan")
        table.add_column("Dataset", style="blue")
        table.add_column("Profile", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Progress", style="green")
        
        for job in filtered_jobs[:20]:  # Limit to last 20
            progress_text = f"{job.percent:.1f}%" if job.percent > 0 else "N/A"
            
            table.add_row(
                job.id,
                job.dataset,
                job.profile,
                _format_job_status_menu(job.status),
                progress_text
            )
        
        console.print(table)
        
        if len(filtered_jobs) > 20:
            console.print(f"[dim]Showing last 20 of {len(filtered_jobs)} jobs[/dim]")
        
        console.print()
        _pause()

# ---------------------------------------------------------------------------
# Enhanced Training Menu (Updated)
# ---------------------------------------------------------------------------

def _training_batch_menu():
    """Enhanced batch training menu with multiple datasets and profiles."""
    _header_with_context("Batch Training", ["AutoTrainV2", "Training", "Batch"])
    
    from autotrain_sdk.paths import BATCH_CONFIG_DIR
    
    # Get available datasets from presets
    available_datasets = set()
    profiles = ["Flux", "FluxLORA", "Nude"]
    
    for profile in profiles:
        presets = list((BATCH_CONFIG_DIR / profile).glob("*.toml"))
        for preset in presets:
            available_datasets.add(preset.stem)
    
    if not available_datasets:
        console.print("[yellow]No datasets with presets found[/yellow]")
        _pause()
        return
    
    # Select datasets
    dataset_choices = [q.Choice(ds, checked=True) for ds in sorted(available_datasets)]
    selected_datasets = q.checkbox("Select datasets for batch training:", choices=dataset_choices).ask()
    
    if not selected_datasets:
        return
    
    # Select profiles
    profile_choices = [q.Choice(profile, checked=True) for profile in profiles]
    selected_profiles = q.checkbox("Select training profiles:", choices=profile_choices).ask()
    
    if not selected_profiles:
        return
    
    # Create job combinations
    job_combinations = []
    for dataset in selected_datasets:
        for profile in selected_profiles:
            preset_path = BATCH_CONFIG_DIR / profile / f"{dataset}.toml"
            if preset_path.exists():
                job_combinations.append((dataset, profile, preset_path))
    
    if not job_combinations:
        console.print("[yellow]No valid preset combinations found[/yellow]")
        _pause()
        return
    
    # Show job plan
    console.print(f"\n[bold cyan]📋 Batch Training Plan ({len(job_combinations)} jobs):[/bold cyan]")
    for i, (dataset, profile, _) in enumerate(job_combinations, 1):
        console.print(f"  {i}. {dataset} → {profile}")
    console.print()
    
    # Confirm execution
    if not q.confirm(f"Queue {len(job_combinations)} training jobs?", default=True).ask():
        return
    
    # Select GPU configuration
    gpu_ids = _select_gpus()
    
    # Execute batch training
    def enqueue_batch_job(job_combo):
        """Encola un job individual del lote"""
        dataset, profile, preset_path = job_combo
        try:
            # Registrar dataset como reciente cuando se entrena en batch
            _add_recent_dataset(dataset)
            
            from autotrain_sdk.paths import compute_run_dir
            run_dir = compute_run_dir(dataset, profile)
            job = Job(dataset, profile, preset_path, run_dir, gpu_ids=gpu_ids)
            _JOB_MANAGER.enqueue(job)
            
            return (True, f"{dataset} ({profile}) → Job {job.id}")
        except Exception as e:
            return (False, f"{dataset} ({profile}) → Error: {str(e)}")
    
    console.print(f"\n[bold cyan]🚀 Starting batch training with {len(job_combinations)} jobs...[/bold cyan]")
    console.print()
    
    # Ejecutar con barra de progreso
    results = _progress_operation(
        items=job_combinations,
        operation_func=enqueue_batch_job,
        title="🎯 Enqueuing batch training jobs",
        item_name="job"
    )
    
    # Mostrar resultados
    console.print("\n[bold blue]📋 Batch Training Results:[/bold blue]")
    success_count = 0
    for success, message in results:
        if success:
            console.print(f"[green]✓ {message}[/green]")
            success_count += 1
        else:
            console.print(f"[red]✗ {message}[/red]")
    
    console.print(f"\n[bold green]✅ Batch training completed! {success_count}/{len(results)} jobs enqueued successfully.[/bold green]")
    _pause()

# ---------------------------------------------------------------------------
# Enhanced Jobs Menu (Updated)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper functions for menu integration (may not exist in gradio_app.py)
# ---------------------------------------------------------------------------

def _get_dataset_info_fallback(dataset_name: str) -> dict:
    """Fallback function to get dataset information if gradio_app function doesn't exist."""
    try:
        from autotrain_sdk.paths import INPUT_DIR
        from autotrain_sdk.dataset_manage import dataset_stats
        
        dataset_path = INPUT_DIR / dataset_name
        if not dataset_path.exists():
            return {"exists": False, "error": f"Dataset '{dataset_name}' not found"}
        
        # Get basic stats
        stats = dataset_stats(dataset_name)
        
        # Calculate total size
        total_size = 0
        for file in dataset_path.glob("**/*"):
            if file.is_file():
                total_size += file.stat().st_size
        
        # Format size
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024**2:
            size_str = f"{total_size/1024:.1f} KB"
        elif total_size < 1024**3:
            size_str = f"{total_size/(1024**2):.1f} MB"
        else:
            size_str = f"{total_size/(1024**3):.1f} GB"
        
        return {
            "exists": True,
            "num_images": stats.get("images", 0),
            "num_captions": stats.get("txt", 0),
            "avg_resolution": f"{stats.get('avg_w', 0)}x{stats.get('avg_h', 0)}",
            "total_size": size_str,
            "last_modified": dataset_path.stat().st_mtime,
            "path": str(dataset_path)
        }
    except Exception as e:
        return {"exists": False, "error": f"Error reading dataset: {str(e)}"}

def _estimate_training_time_fallback(dataset_name: str, profile: str) -> dict:
    """Fallback function to estimate training time if gradio_app function doesn't exist."""
    try:
        from autotrain_sdk.paths import BATCH_CONFIG_DIR
        from autotrain_sdk.configurator import load_config
        
        preset_path = BATCH_CONFIG_DIR / profile / f"{dataset_name}.toml"
        if not preset_path.exists():
            return {"error": f"Preset not found for {dataset_name} ({profile})"}
        
        config = load_config(preset_path)
        
        # Get basic parameters
        epochs = config.get("max_train_epochs", 1)
        batch_size = config.get("train_batch_size", 1)
        repeats = config.get("dataset_repeats", 30)
        
        # Get dataset info
        dataset_info = _get_dataset_info_fallback(dataset_name)
        if not dataset_info.get("exists", False):
            return {"error": "Dataset not found"}
        
        num_images = dataset_info.get("num_images", 0)
        if num_images == 0:
            return {"error": "No images found in dataset"}
        
        # Simple time estimation
        total_steps = (num_images * repeats * epochs) // batch_size
        
        # Rough estimate: 1 step per second (very rough)
        estimated_seconds = total_steps * 1.5  # 1.5 seconds per step
        
        # Format time
        if estimated_seconds < 60:
            time_str = f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            time_str = f"{estimated_seconds/60:.0f} minutes"
        else:
            time_str = f"{estimated_seconds/3600:.1f} hours"
        
        return {
            "estimated_time": time_str,
            "total_steps": total_steps,
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "repeats": repeats
            }
        }
    except Exception as e:
        return {"error": f"Error estimating training time: {str(e)}"}

def _format_enhanced_metrics_fallback(metrics_data: dict) -> str:
    """Fallback function to format enhanced metrics if gradio_app function doesn't exist."""
    if not metrics_data:
        return "No metrics data available"
    
    lines = []
    
    # Basic metrics
    if "loss" in metrics_data:
        lines.append(f"📉 Loss: {metrics_data['loss']:.4f}")
    
    if "lr" in metrics_data:
        lines.append(f"📚 Learning Rate: {metrics_data['lr']:.6f}")
    
    if "step" in metrics_data:
        lines.append(f"📊 Step: {metrics_data['step']}")
    
    if "epoch" in metrics_data:
        lines.append(f"🔄 Epoch: {metrics_data['epoch']}")
    
    return "\n".join(lines) if lines else "No metrics available"

# ---------------------------------------------------------------------------
# Updated helper functions to use fallbacks
# ---------------------------------------------------------------------------

def _create_system_stats_panel():
    """Return a Rich Panel with up-to-date system stats summary (usable inside Live refresh)."""
    try:
        stats = _get_system_stats()
    except Exception:
        # Fallback empty panel on error
        return Panel("[dim]Stats unavailable[/dim]", border_style="dim blue")

    parts: list[str] = []
    if stats.get("input_datasets", 0):
        parts.append(f"Datasets: {stats['input_datasets']}")
    if stats.get("pending_datasets", 0):
        parts.append(f"Pending datasets: {stats['pending_datasets']}")
    if stats.get("presets", 0):
        parts.append(f"Presets: {stats['presets']}")

    # Jobs info
    running = stats.get("running_jobs", 0)
    pending = stats.get("pending_jobs", 0)
    if running or pending:
        if running and pending:
            parts.append(f"Jobs: {running} running, {pending} queued")
        elif running:
            parts.append(f"Jobs: {running} running")
        else:
            parts.append(f"Jobs: {pending} queued")

    summary = " • ".join(parts) if parts else "No stats"  # Graceful fallback
    return Panel(summary, style="dim", border_style="dim blue", padding=(0, 1))

# ---------------------------------------------------------------------------
# Upcoming jobs table (for Live Monitor)
# ---------------------------------------------------------------------------

def _create_upcoming_jobs_table(limit: int = 6) -> Table | None:
    """Return a Table with the next pending jobs after the current one.

    If there are no pending jobs, returns None so caller can skip rendering.
    """

    jobs = _JOB_MANAGER.list_jobs()
    pending_jobs = [j for j in jobs if j.status == JobStatus.PENDING]

    if not pending_jobs:
        return None

    table = Table(
        show_header=True,
        header_style="dim",
        box=None,
        title="📅 Upcoming Jobs",
    )
    table.add_column("Job", style="cyan", width=8)
    table.add_column("Dataset", style="blue", width=12)
    table.add_column("Profile", style="magenta", width=10)
    table.add_column("ETA", style="yellow", width=8)

    for idx, job in enumerate(pending_jobs[:limit], 1):
        # Rough ETA: assuming ~2h per job position (same as in compact table)
        eta_text = f"~{idx*2}h"
        table.add_row(job.id, job.dataset[:12], job.profile, eta_text)

    if len(pending_jobs) > limit:
        table.add_row("…", f"+{len(pending_jobs)-limit} more", "", "")

    return table

# ---------------------------------------------------------------------------
# Smart Queue Management Detection
# ---------------------------------------------------------------------------

def _should_use_optimized_view(jobs: List[Job]) -> bool:
    """Determine if optimized view should be used based on queue size and composition."""
    total_jobs = len(jobs)
    
    # Use optimized view for large queues
    if total_jobs > 100:
        return True
    
    # Also use optimized view if there are many completed jobs (performance impact)
    completed_jobs = len([j for j in jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}])
    if completed_jobs > 50:
        return True
    
    return False

def _show_queue_performance_warning(jobs_count: int):
    """Show performance warning for large queues with optimization suggestions."""
    console.print(f"[yellow]⚠️ Large queue detected ({jobs_count} jobs)[/yellow]")
    console.print()
    
    if jobs_count > 1000:
        console.print("[red]🚨 Very large queue (>1000 jobs) - performance may be significantly impacted[/red]")
        console.print("[dim]Recommendations:[/dim]")
        console.print("[dim]  • Use optimized view for better performance[/dim]")
        console.print("[dim]  • Clean up old completed jobs periodically[/dim]")
        console.print("[dim]  • Consider archiving historical jobs[/dim]")
    elif jobs_count > 500:
        console.print("[yellow]⚡ Large queue (>500 jobs) - using optimized view recommended[/yellow]")
        console.print("[dim]  • Optimized view provides better navigation and performance[/dim]")
    else:
        console.print("[cyan]ℹ️ Moderate queue size - optimized view available for better navigation[/cyan]")
    
    console.print()

# ---------------------------------------------------------------------------
# Enhanced Job Management for Large Queues (1000+ jobs optimization)
# ---------------------------------------------------------------------------

class JobFilter(str, Enum):
    """Job filter options for large queue management."""
    ALL = "all"
    ACTIVE = "active"  # running + pending
    RUNNING = "running"
    PENDING = "pending"
    COMPLETED = "completed"  # done + failed + canceled
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"
    RECENT = "recent"  # last 24 hours

def _get_filtered_jobs(jobs: List[Job], filter_type: JobFilter = JobFilter.ALL, 
                      search_term: str = "", limit: int = 0) -> List[Job]:
    """Get filtered and optionally limited list of jobs for performance."""
    
    # Apply status filter first (most selective)
    if filter_type == JobFilter.ACTIVE:
        filtered = [j for j in jobs if j.status in {JobStatus.RUNNING, JobStatus.PENDING}]
    elif filter_type == JobFilter.RUNNING:
        filtered = [j for j in jobs if j.status == JobStatus.RUNNING]
    elif filter_type == JobFilter.PENDING:
        filtered = [j for j in jobs if j.status == JobStatus.PENDING]
    elif filter_type == JobFilter.COMPLETED:
        filtered = [j for j in jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
    elif filter_type == JobFilter.DONE:
        filtered = [j for j in jobs if j.status == JobStatus.DONE]
    elif filter_type == JobFilter.FAILED:
        filtered = [j for j in jobs if j.status == JobStatus.FAILED]
    elif filter_type == JobFilter.CANCELED:
        filtered = [j for j in jobs if j.status == JobStatus.CANCELED]
    elif filter_type == JobFilter.RECENT:
        # Recent jobs (simple heuristic: jobs with recent IDs)
        # Sort by ID and take latest (since IDs contain timestamps)
        sorted_jobs = sorted(jobs, key=lambda x: x.id, reverse=True)
        filtered = sorted_jobs[:100]  # Last 100 as "recent"
    else:  # ALL
        filtered = jobs
    
    # Apply search filter if provided
    if search_term:
        search_lower = search_term.lower()
        filtered = [j for j in filtered if (
            search_lower in j.id.lower() or
            search_lower in j.dataset.lower() or
            search_lower in j.profile.lower()
        )]
    
    # Apply limit if specified
    if limit > 0:
        filtered = filtered[:limit]
    
    return filtered

def _get_job_stats_fast(jobs: List[Job]) -> Dict[str, int]:
    """Get job statistics efficiently without multiple iterations."""
    stats = {
        "total": len(jobs),
        "running": 0,
        "pending": 0,
        "done": 0,
        "failed": 0,
        "canceled": 0
    }
    
    for job in jobs:
        if job.status == JobStatus.RUNNING:
            stats["running"] += 1
        elif job.status == JobStatus.PENDING:
            stats["pending"] += 1
        elif job.status == JobStatus.DONE:
            stats["done"] += 1
        elif job.status == JobStatus.FAILED:
            stats["failed"] += 1
        elif job.status == JobStatus.CANCELED:
            stats["canceled"] += 1
    
    stats["active"] = stats["running"] + stats["pending"]
    stats["completed"] = stats["done"] + stats["failed"] + stats["canceled"]
    
    return stats

def _create_optimized_job_table(jobs: List[Job], page_size: int = 15, page: int = 1, 
                                filter_type: JobFilter = JobFilter.ALL, 
                                search_term: str = "") -> Tuple[Table, Dict[str, Any]]:
    """Create an optimized job table with pagination and filtering (15 jobs per page)."""
    
    # Get filtered jobs
    filtered_jobs = _get_filtered_jobs(jobs, filter_type, search_term)
    
    # Calculate pagination
    total_filtered = len(filtered_jobs)
    total_pages = (total_filtered + page_size - 1) // page_size if total_filtered > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_filtered)
    page_jobs = filtered_jobs[start_idx:end_idx]
    
    # Create table with optimized rendering
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Job", style="cyan", width=10)
    table.add_column("Dataset", style="blue", width=15)
    table.add_column("Status", style="bold", width=12)
    table.add_column("Progress", style="green", width=15)
    table.add_column("ETA", style="yellow", width=10)
    
    if not page_jobs:
        table.add_row("—", "—", "—", "No jobs found", "—")
    else:
        for job in page_jobs:
            # Format status with emoji (cached for performance)
            status_text = _format_job_status_menu(job.status)
            
            # Format progress (optimized)
            if job.status == JobStatus.RUNNING and job.progress_str:
                progress_text = f"{job.percent:.0f}%" if job.percent else job.progress_str
            elif job.status == JobStatus.PENDING:
                # Simplified position calculation (avoid expensive iteration)
                progress_text = "Queued"
            elif job.status == JobStatus.DONE:
                progress_text = "100%"
            elif job.status == JobStatus.FAILED:
                progress_text = "Failed"
            elif job.status == JobStatus.CANCELED:
                progress_text = "Canceled"
            else:
                progress_text = "—"
            
            # Format ETA (simplified)
            if job.status == JobStatus.RUNNING and job.eta:
                eta_text = job.eta
            elif job.status == JobStatus.DONE:
                eta_text = "Complete"
            else:
                eta_text = "—"
            
            table.add_row(
                job.id[:10] + "..." if len(job.id) > 10 else job.id,
                job.dataset[:15] + "..." if len(job.dataset) > 15 else job.dataset,
                status_text,
                progress_text,
                eta_text
            )
    
    # Create metadata for pagination info
    metadata = {
        "total_jobs": len(jobs),
        "filtered_jobs": total_filtered,
        "current_page": page,
        "total_pages": total_pages,
        "page_size": page_size,
        "showing_from": start_idx + 1 if page_jobs else 0,
        "showing_to": end_idx,
        "filter_type": filter_type,
        "search_term": search_term
    }
    
    return table, metadata

def _show_pagination_info(metadata: Dict[str, Any]) -> None:
    """Show pagination and filtering information."""
    total = metadata["total_jobs"]
    filtered = metadata["filtered_jobs"]
    page = metadata["current_page"]
    total_pages = metadata["total_pages"]
    showing_from = metadata["showing_from"]
    showing_to = metadata["showing_to"]
    filter_type = metadata["filter_type"]
    search_term = metadata["search_term"]
    
    # Filter info
    filter_info = f"Filter: {filter_type.value}"
    if search_term:
        filter_info += f" | Search: '{search_term}'"
    
    # Pagination info
    if filtered > 0:
        page_info = f"Showing {showing_from}-{showing_to} of {filtered}"
        if filtered != total:
            page_info += f" (filtered from {total} total)"
        page_info += f" | Page {page}/{total_pages}"
    else:
        page_info = f"No jobs found (total: {total})"
    
    console.print(f"[dim]{filter_info}[/dim]")
    console.print(f"[dim]{page_info}[/dim]")
    console.print()

def _jobs_menu_optimized():
    """Optimized jobs menu for handling large queues (1000+ jobs)."""
    
    # Configuration
    page_size = 15  # Jobs per page
    current_page = 1
    current_filter = JobFilter.ACTIVE  # Start with active jobs (most relevant)
    search_term = ""
    
    while True:
        _header_with_context("Job Management (Optimized)", ["AutoTrainV2", "Jobs"])
        
        # Get all jobs once (but don't process them all)
        all_jobs = _JOB_MANAGER.list_jobs()
        
        # Get fast statistics
        stats = _get_job_stats_fast(all_jobs)
        
        # Show quick stats
        if stats["total"] > 0:
            console.print(f"[bold]📊 Queue Overview:[/bold] {stats['total']} total | "
                         f"🔄 {stats['running']} running | ⏸️ {stats['pending']} pending | "
                         f"✅ {stats['done']} completed | ❌ {stats['failed']} failed")
            
            if stats["total"] > 100:
                console.print(f"[yellow]⚡ Large queue detected ({stats['total']} jobs) - using optimized view[/yellow]")
            console.print()
        
        # Create optimized table
        table, metadata = _create_optimized_job_table(
            all_jobs, page_size, current_page, current_filter, search_term
        )
        
        # Show pagination info
        _show_pagination_info(metadata)
        
        # Show table
        console.print(table)
        console.print()
        
        # Navigation options
        nav_choices = []
        
        # Filter options
        nav_choices.extend([
            q.Separator("╔══════════════════╗"),
            q.Separator("║  📋 FILTERING    ║"),
            q.Separator("╚══════════════════╝"),
            q.Choice("f. Change filter", "filter"),
            q.Choice("s. Search jobs", "search"),
            q.Choice("c. Clear search", "clear_search"),
        ])
        
        # Pagination options
        if metadata["total_pages"] > 1:
            nav_choices.extend([
                q.Separator("╔═══════════════════╗"),
                q.Separator("║  📄 NAVIGATION    ║"),
                q.Separator("╚═══════════════════╝"),
            ])
            
            if current_page > 1:
                nav_choices.append(q.Choice("p. Previous page", "prev"))
            if current_page < metadata["total_pages"]:
                nav_choices.append(q.Choice("n. Next page", "next"))
            
            nav_choices.extend([
                q.Choice("g. Go to page", "goto"),
                q.Choice("↑. First page", "first"),
                q.Choice("↓. Last page", "last"),
            ])
        
        # Management options
        nav_choices.extend([
            q.Separator("╔═══════════════════╗"),
            q.Separator("║  🔧 MANAGEMENT    ║"),
            q.Separator("╚═══════════════════╝"),
            q.Choice("1. Job details", "details"),
            q.Choice("2. Cancel/Remove jobs", "cancel"),
            q.Choice("3. Live monitor", "monitor"),
            q.Choice("4. Bulk operations", "bulk"),
        ])
        
        # System options
        nav_choices.extend([
            q.Separator("╔═══════════════════╗"),
            q.Separator("║  🔄 SYSTEM        ║"),
            q.Separator("╚═══════════════════╝"),
            q.Choice("r. Refresh", "refresh"),
            q.Choice("t. Toggle view mode", "toggle"),
            q.Choice("0. Back to main menu", "back"),
        ])
        
        action = q.select("Choose action:", choices=nav_choices).ask()
        
        if action in {None, "back"}:
            return
        elif action == "filter":
            new_filter = q.select(
                "Select filter:",
                choices=[
                    q.Choice("Active (Running + Pending)", JobFilter.ACTIVE),
                    q.Choice("All jobs", JobFilter.ALL),
                    q.Choice("Running only", JobFilter.RUNNING),
                    q.Choice("Pending only", JobFilter.PENDING),
                    q.Choice("Completed (Done + Failed + Canceled)", JobFilter.COMPLETED),
                    q.Choice("Done only", JobFilter.DONE),
                    q.Choice("Failed only", JobFilter.FAILED),
                    q.Choice("Canceled only", JobFilter.CANCELED),
                    q.Choice("Recent (last 100)", JobFilter.RECENT),
                ],
                default=current_filter
            ).ask()
            if new_filter:
                current_filter = new_filter
                current_page = 1  # Reset to first page
        elif action == "search":
            new_search = q.text("Search term (ID, dataset, profile):").ask()
            if new_search is not None:
                search_term = new_search.strip()
                current_page = 1  # Reset to first page
        elif action == "clear_search":
            search_term = ""
            current_page = 1
        elif action == "prev":
            current_page = max(1, current_page - 1)
        elif action == "next":
            current_page = min(metadata["total_pages"], current_page + 1)
        elif action == "goto":
            try:
                page_num = int(q.text(f"Go to page (1-{metadata['total_pages']}):").ask() or "0")
                if 1 <= page_num <= metadata["total_pages"]:
                    current_page = page_num
                else:
                    console.print(f"[red]Invalid page number. Use 1-{metadata['total_pages']}[/red]")
                    _pause()
            except ValueError:
                console.print("[red]Invalid page number[/red]")
                _pause()
        elif action == "first":
            current_page = 1
        elif action == "last":
            current_page = metadata["total_pages"]
        elif action == "refresh":
            continue  # Just refresh the view
        elif action == "details":
            _job_details_optimized(all_jobs, current_filter, search_term)
        elif action == "cancel":
            _cancel_remove_jobs_optimized(all_jobs, current_filter, search_term)
        elif action == "monitor":
            _training_monitor_menu()
        elif action == "bulk":
            _bulk_operations_menu(all_jobs, current_filter, search_term)
        elif action == "toggle":
            # Toggle between optimized and legacy view
            console.print("[cyan]Switching to legacy view...[/cyan]")
            time.sleep(1)
            _view_queue_compact()
            break

def _job_details_optimized(all_jobs: List[Job], filter_type: JobFilter, search_term: str):
    """Show job details with optimized selection for large queues."""
    filtered_jobs = _get_filtered_jobs(all_jobs, filter_type, search_term, limit=30)  # Limit for UI
    
    if not filtered_jobs:
        console.print("[yellow]No jobs found with current filter[/yellow]")
        _pause()
        return
    
    # If too many jobs, use search
    if len(filtered_jobs) > 15:
        job_id = q.text("Enter Job ID (or partial ID):").ask()
        if not job_id:
            return
        
        matching_jobs = [j for j in filtered_jobs if job_id.lower() in j.id.lower()]
        if not matching_jobs:
            console.print(f"[yellow]No jobs found matching '{job_id}'[/yellow]")
            _pause()
            return
        elif len(matching_jobs) == 1:
            selected_job = matching_jobs[0]
        else:
            # Multiple matches, let user select
            choices = [f"{j.id} ({j.dataset}, {j.status.value})" for j in matching_jobs[:15]]
            selected = q.select("Multiple matches found:", choices=choices + ["Cancel"]).ask()
            if not selected or selected == "Cancel":
                return
            job_id = selected.split(" ")[0]
            selected_job = next((j for j in matching_jobs if j.id == job_id), None)
    else:
        # Small list, show selection
        choices = [f"{j.id} ({j.dataset}, {j.status.value})" for j in filtered_jobs]
        selected = q.select("Select job for details:", choices=choices + ["Cancel"]).ask()
        if not selected or selected == "Cancel":
            return
        job_id = selected.split(" ")[0]
        selected_job = next((j for j in filtered_jobs if j.id == job_id), None)
    
    if selected_job:
        _show_job_details_menu(selected_job)

def _cancel_remove_jobs_optimized(all_jobs: List[Job], filter_type: JobFilter, search_term: str):
    """Optimized cancel/remove interface for large queues."""
    _header_with_context("Cancel/Remove Jobs (Optimized)", ["AutoTrainV2", "Jobs", "Cancel"])
    
    # Show quick stats
    stats = _get_job_stats_fast(all_jobs)
    console.print(f"[dim]📊 {stats['running']} running | {stats['pending']} pending | {stats['completed']} completed[/dim]")
    console.print()
    
    action = q.select(
        "Choose bulk operation:",
        choices=[
            q.Choice("Cancel specific job (by ID)", "cancel_specific"),
            q.Choice("Remove specific job (by ID)", "remove_specific"),
            q.Choice("Cancel all running jobs", "cancel_all_running"),
            q.Choice("Remove all completed jobs", "remove_all_completed"),
            q.Choice("Emergency stop all active jobs", "emergency_stop"),
            q.Choice("Back", "back")
        ]
    ).ask()
    
    if action in {None, "back"}:
        return
    elif action == "cancel_specific":
        job_id = q.text("Enter Job ID to cancel:").ask()
        if job_id and q.confirm(f"Cancel job {job_id}?", default=False).ask():
            _JOB_MANAGER.cancel(job_id)
            console.print(f"[green]✅ Job {job_id} canceled[/green]")
            _pause()
    elif action == "remove_specific":
        job_id = q.text("Enter Job ID to remove:").ask()
        if job_id and q.confirm(f"Remove job {job_id}?", default=False).ask():
            if _JOB_MANAGER.remove_job(job_id):
                console.print(f"[green]✅ Job {job_id} removed[/green]")
            else:
                console.print(f"[yellow]⚠️ Could not remove job {job_id}[/yellow]")
            _pause()
    elif action == "cancel_all_running":
        running_jobs = [j for j in all_jobs if j.status == JobStatus.RUNNING]
        if not running_jobs:
            console.print("[yellow]No running jobs to cancel[/yellow]")
            _pause()
            return
        
        console.print(f"[red]⚠️ This will cancel {len(running_jobs)} running job(s)[/red]")
        if q.confirm("Cancel all running jobs?", default=False).ask():
            console.print(f"[cyan]Canceling {len(running_jobs)} running jobs...[/cyan]")
            
            # Use progress bar for large operations (though running jobs are usually few)
            if len(running_jobs) > 5:
                with _create_progress_bar("Canceling Running Jobs") as progress:
                    task = progress.add_task("[red]Canceling running jobs...", total=len(running_jobs))
                    for i, job in enumerate(running_jobs, 1):
                        _JOB_MANAGER.cancel(job.id)
                        progress.update(task, description=f"[red]Canceled {i}/{len(running_jobs)} running jobs")
                        progress.advance(task)
            else:
                # For small operations, show individual progress
                for i, job in enumerate(running_jobs, 1):
                    console.print(f"[dim]Canceling running job {i}/{len(running_jobs)}: {job.id}[/dim]")
                    _JOB_MANAGER.cancel(job.id)
            
            console.print(f"[green]✅ Canceled {len(running_jobs)} running jobs[/green]")
            _pause()
    elif action == "remove_all_completed":
        completed_jobs = [j for j in all_jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
        if not completed_jobs:
            console.print("[yellow]No completed jobs to remove[/yellow]")
            _pause()
            return
        
        console.print(f"[yellow]⚠️ This will remove {len(completed_jobs)} completed job(s)[/yellow]")
        if q.confirm("Remove all completed jobs?", default=False).ask():
            console.print(f"[cyan]Removing {len(completed_jobs)} completed jobs...[/cyan]")
            
            removed = 0
            # Use progress bar for large operations
            if len(completed_jobs) > 10:
                with _create_progress_bar("Removing Jobs") as progress:
                    task = progress.add_task("[yellow]Removing completed jobs...", total=len(completed_jobs))
                    for i, job in enumerate(completed_jobs, 1):
                        if _JOB_MANAGER.remove_job(job.id):
                            removed += 1
                        progress.update(task, description=f"[yellow]Removed {removed}/{len(completed_jobs)} jobs")
                        progress.advance(task)
            else:
                # For small operations, show individual progress
                for i, job in enumerate(completed_jobs, 1):
                    console.print(f"[dim]Removing job {i}/{len(completed_jobs)}: {job.id}[/dim]")
                    if _JOB_MANAGER.remove_job(job.id):
                        removed += 1
            
            console.print(f"[green]✅ Removed {removed} completed jobs[/green]")
            _pause()
    elif action == "emergency_stop":
        active_jobs = [j for j in all_jobs if j.status in {JobStatus.RUNNING, JobStatus.PENDING}]
        if not active_jobs:
            console.print("[yellow]No active jobs to stop[/yellow]")
            _pause()
            return
        
        console.print(f"[red]🚨 EMERGENCY STOP: This will cancel {len(active_jobs)} active job(s)[/red]")
        if q.confirm("Emergency stop all active jobs?", default=False).ask():
            console.print(f"[cyan]Canceling {len(active_jobs)} jobs...[/cyan]")
            
            # Use progress bar for large operations
            if len(active_jobs) > 10:
                with _create_progress_bar("Emergency Stop") as progress:
                    task = progress.add_task("[red]Canceling jobs...", total=len(active_jobs))
                    for i, job in enumerate(active_jobs, 1):
                        _JOB_MANAGER.cancel(job.id)
                        progress.update(task, description=f"[red]Canceled {i}/{len(active_jobs)} jobs")
                        progress.advance(task)
            else:
                # For small operations, show individual progress
                for i, job in enumerate(active_jobs, 1):
                    console.print(f"[dim]Canceling job {i}/{len(active_jobs)}: {job.id}[/dim]")
                    _JOB_MANAGER.cancel(job.id)
            
            console.print(f"[green]✅ Emergency stop completed for {len(active_jobs)} jobs[/green]")
            console.print("[dim]Note: Integration notifications are optimized to avoid spam when disabled[/dim]")
            _pause()

def _bulk_operations_menu(all_jobs: List[Job], filter_type: JobFilter, search_term: str):
    """Bulk operations menu for managing large job queues."""
    _header_with_context("Bulk Operations", ["AutoTrainV2", "Jobs", "Bulk"])
    
    stats = _get_job_stats_fast(all_jobs)
    console.print(f"[bold]📊 Queue Statistics:[/bold]")
    console.print(f"  Total: {stats['total']} | Active: {stats['active']} | Completed: {stats['completed']}")
    
    # Show integration status to inform user about potential notifications
    gsheets_enabled = os.getenv("AUTO_GSHEET_ENABLE", "0") == "1"
    console.print(f"[dim]🔧 Google Sheets notifications: {'enabled' if gsheets_enabled else 'disabled (no warnings)'}[/dim]")
    console.print()
    
    action = q.select(
        "Choose bulk operation:",
        choices=[
            q.Choice("Clean old completed jobs (keep last 50)", "clean_old"),
            q.Choice("Archive completed jobs to file", "archive"),
            q.Choice("Export job list to CSV", "export"),
            q.Choice("Show detailed statistics", "detailed_stats"),
            q.Choice("Optimize queue performance", "optimize"),
            q.Choice("Back", "back")
        ]
    ).ask()
    
    if action in {None, "back"}:
        return
    elif action == "clean_old":
        # Keep only last 50 completed jobs
        completed_jobs = [j for j in all_jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
        if len(completed_jobs) <= 50:
            console.print("[green]Queue is already optimized (≤50 completed jobs)[/green]")
            _pause()
            return
        
        # Sort by job ID (which contains timestamp) and keep newest 50
        sorted_completed = sorted(completed_jobs, key=lambda x: x.id, reverse=True)
        jobs_to_remove = sorted_completed[50:]  # Remove oldest
        
        console.print(f"[yellow]This will remove {len(jobs_to_remove)} old completed jobs (keeping newest 50)[/yellow]")
        if q.confirm("Continue with cleanup?", default=True).ask():
            removed = 0
            for job in jobs_to_remove:
                if _JOB_MANAGER.remove_job(job.id):
                    removed += 1
            console.print(f"[green]✅ Cleaned up {removed} old jobs[/green]")
            _pause()
    elif action == "detailed_stats":
        _show_detailed_job_statistics(all_jobs)
    elif action == "optimize":
        console.print("[cyan]🔧 Optimizing job queue performance...[/cyan]")
        
        # Clear completed jobs older than certain threshold
        completed_count = stats['completed']
        if completed_count > 100:
            console.print(f"[yellow]Found {completed_count} completed jobs (recommended: keep <100)[/yellow]")
            if q.confirm("Remove completed jobs to improve performance?", default=True).ask():
                removed = _JOB_MANAGER.clear_completed_jobs()
                console.print(f"[green]✅ Removed {removed} completed jobs[/green]")
        
        console.print("[green]✅ Queue optimization completed[/green]")
        _pause()
    
    # Add other bulk operations as needed...

def _show_detailed_job_statistics(all_jobs: List[Job]):
    """Show detailed statistics about the job queue."""
    _header_with_context("Detailed Job Statistics", ["AutoTrainV2", "Jobs", "Statistics"])
    
    stats = _get_job_stats_fast(all_jobs)
    
    # Basic stats table
    stats_table = Table("Metric", "Count", "Percentage", title="📊 Job Queue Statistics")
    total = stats["total"]
    
    for status, count in [
        ("Total Jobs", stats["total"]),
        ("Running", stats["running"]),
        ("Pending", stats["pending"]),
        ("Completed", stats["done"]),
        ("Failed", stats["failed"]),
        ("Canceled", stats["canceled"]),
    ]:
        percentage = f"{(count/total*100):.1f}%" if total > 0 else "0%"
        stats_table.add_row(status, str(count), percentage)
    
    console.print(stats_table)
    console.print()
    
    # Dataset statistics
    dataset_counts = {}
    profile_counts = {}
    
    for job in all_jobs:
        dataset_counts[job.dataset] = dataset_counts.get(job.dataset, 0) + 1
        profile_counts[job.profile] = profile_counts.get(job.profile, 0) + 1
    
    # Top datasets
    top_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_datasets:
        dataset_table = Table("Dataset", "Jobs", title="📂 Top Datasets")
        for dataset, count in top_datasets:
            dataset_table.add_row(dataset, str(count))
        console.print(dataset_table)
        console.print()
    
    # Profile distribution
    profile_table = Table("Profile", "Jobs", title="⚙️ Profile Distribution")
    for profile, count in sorted(profile_counts.items(), key=lambda x: x[1], reverse=True):
        profile_table.add_row(profile, str(count))
    console.print(profile_table)
    console.print()
    
    _pause()

# ---------------------------------------------------------------------------
# Integration with existing menu system
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user[/red]")
        sys.exit(1) 
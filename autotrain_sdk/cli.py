from __future__ import annotations

"""Unified CLI for AutoTrainV2 (Phase 2).

Examples::

    # Dataset Management
    python -m autotrain_sdk dataset create --names "alex,maria"
    autotrain dataset build-output --min-images 20
    autotrain dataset create-prompts
    
    # Configuration Management
    autotrain config refresh
    autotrain config generate --dataset alex
    
    # Training Commands
    autotrain train start --profile Flux --file BatchConfig/Flux/alex.toml
    autotrain train info alex --profile Flux
    autotrain train interactive
    autotrain train batch --datasets "alex,maria" --profile FluxLORA
    
    # Training Monitoring & Management
    autotrain train monitor
    autotrain train status --verbose
    autotrain train queue list
    autotrain train queue cancel <job_id>
    autotrain train logs --follow
    autotrain train stop --job <job_id>
    autotrain train history --limit 5
    autotrain train clean --status done
    
    # Pipeline Commands
    autotrain pipeline run --dataset-path /path/to/dataset --profile Flux --monitor
    autotrain pipeline prepare --dataset-path /path/to/dataset --profile FluxLORA
    autotrain pipeline batch --datasets-dir /path/to/datasets --profile FluxLORA --monitor
    autotrain pipeline status
    
    # Web Interface
    autotrain web
    autotrain web --port 8080
"""

import shutil
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional

import typer
import rich_click as click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
import os  # NEW: for environment variable checks

from .dataset import (
    create_input_folders,
    populate_output_structure,
    clean_workspace,
    create_sample_prompts,
    create_sample_prompts_for_dataset,
)
from .configurator import (
    load_config,
    update_config,
    save_config,
    generate_presets,
)
from .trainer import run_training
from .gradio_app import launch as launch_web, _get_dataset_info, _estimate_training_time, _format_enhanced_metrics
from .job_manager import JOB_MANAGER, Job, JobStatus
from .paths import BATCH_CONFIG_DIR, compute_run_dir
from .sweeps import parse_grid, expand_grid, generate_variant

console = Console()
app = typer.Typer(add_completion=False, rich_help_panel="AutoTrain")

# ---------------------------------------------------------------------------
# Helper Functions for Enhanced Training CLI
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def _format_job_status(status: JobStatus) -> str:
    """Format job status with colored emoji."""
    status_map = {
        JobStatus.RUNNING: "🟢 RUNNING",
        JobStatus.PENDING: "⏸️ PENDING", 
        JobStatus.DONE: "✅ DONE",
        JobStatus.FAILED: "❌ FAILED",
        JobStatus.CANCELED: "🚫 CANCELED"
    }
    return status_map.get(status, str(status))

def _create_dataset_info_table(dataset_name: str, profile: str) -> Table:
    """Create a rich table with dataset information."""
    dataset_info = _get_dataset_info(dataset_name)
    
    table = Table(title=f"Dataset Information: {dataset_name}", show_header=False)
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

def _create_training_estimation_table(dataset_name: str, profile: str) -> Table:
    """Create a rich table with training time estimation."""
    estimation = _estimate_training_time(dataset_name, profile)
    
    table = Table(title=f"Training Estimation: {profile}", show_header=False)
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

def _create_queue_table() -> Table:
    """Create a rich table showing the training queue."""
    jobs = JOB_MANAGER.list_jobs()
    
    table = Table(title="Training Queue")
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
            _format_job_status(job.status),
            progress_text
        )
    
    if not jobs:
        table.add_row("", "", "", "", "No jobs in queue")
    
    return table

def _create_metrics_panel(job: Job) -> Panel:
    """Create a panel with enhanced metrics for monitoring."""
    metrics_data = JOB_MANAGER.get_metrics_data(job.id)
    
    if not metrics_data:
        content = Text("⏳ Waiting for training metrics...", style="yellow")
        return Panel(content, title="📊 Enhanced Metrics", border_style="blue")
    
    # Format metrics using the existing function
    metrics_text = _format_enhanced_metrics(metrics_data)
    
    # Convert markdown-style formatting to rich text
    content = Text()
    for line in metrics_text.split('\n'):
        if line.startswith('**') and line.endswith('**'):
            # Headers
            content.append(line.replace('**', ''), style="bold blue")
        elif line.startswith('• **') and ':**' in line:
            # Metrics entries
            parts = line.split(':**', 1)
            label = parts[0].replace('• **', '  • ')
            value = parts[1].strip() if len(parts) > 1 else ''
            content.append(label, style="cyan")
            content.append(': ')
            content.append(value, style="white")
        elif line.strip():
            content.append(line)
        content.append('\n')
    
    return Panel(content, title="📊 Enhanced Metrics", border_style="blue")

def _create_log_panel(job: Job) -> Panel:
    """Create a panel with recent training log."""
    log, _ = JOB_MANAGER.get_live_output(job.id)
    
    if not log:
        content = Text("⏳ Waiting for training log...", style="yellow")
    else:
        # Show last 10 lines
        lines = log.split('\n')[-10:]
        content = Text('\n'.join(lines), style="white")
    
    return Panel(content, title="📝 Training Log", border_style="red")

def _create_status_panel(job: Job) -> Panel:
    """Create a panel with job status information."""
    content = Text()
    content.append("📊 Status: ", style="bold")
    content.append(f"{job.status.upper()}\n", style="green" if job.status == JobStatus.RUNNING else "white")
    
    if job.progress_str:
        content.append("📈 Progress: ", style="bold")
        content.append(f"{job.progress_str}\n", style="yellow")
    
    if job.avg_loss is not None:
        content.append("📉 Loss: ", style="bold") 
        content.append(f"{job.avg_loss:.4f}\n", style="cyan")
    
    content.append("🔥 Job: ", style="bold")
    content.append(f"{job.id} ({job.dataset})", style="magenta")
    
    return Panel(content, title="ℹ️ Status", border_style="green")

# ---------------------------------------------------------------------------
# App and Sub-app Definitions
# ---------------------------------------------------------------------------

dataset_app = typer.Typer(help="Dataset management")
config_app = typer.Typer(help="TOML preset management")
train_app = typer.Typer(help="Run trainings")

# New sub-app for caption generation
caption_app = typer.Typer(help="Generate captions for images using Florence-2")

app.add_typer(dataset_app, name="dataset")
app.add_typer(config_app, name="config")
app.add_typer(train_app, name="train")
app.add_typer(caption_app, name="caption")

# add web subcommand
a_pp = typer.Typer(help="Launch Gradio web interface")
app.add_typer(a_pp, name="web")

# Integrations subcommands
int_app = typer.Typer(help="Configure external integrations (HF, S3, Google Sheets, …)")
app.add_typer(int_app, name="integrations")

# Registry commands
registry_app = typer.Typer(help="Manage run registry")
app.add_typer(registry_app, name="registry")

# --------------------------------------------------------------
# Pipeline commands
# --------------------------------------------------------------

pipeline_app = typer.Typer(help="Automated training pipeline")
app.add_typer(pipeline_app, name="pipeline")

# --------------------------------------------------------------
# Sweep commands
# --------------------------------------------------------------

sweep_app = typer.Typer(help="Generate grid-search presets for a dataset")
app.add_typer(sweep_app, name="sweep")


@sweep_app.command("create", help="Create presets expanding a parameter grid")
def sweep_create(
    dataset: str = typer.Option(..., help="Dataset folder name, e.g. 'b09g13'"),
    profile: str = typer.Option(..., help="Profile: Flux | FluxLORA | Nude"),
    param: list[str] = typer.Option(..., help="key=v1,v2 (repeat flag for multiple keys)"),
):
    """Generate one TOML file per combination of the provided parameter grid."""

    try:
        grid = parse_grid(param)
    except ValueError as e:
        typer.echo(f"[red]ERROR[/red] {e}")
        raise typer.Exit(code=1)

    combos = expand_grid(grid)
    typer.echo(f"Generating {len(combos)} preset(s)…")

    created = []
    for overrides in combos:
        path = generate_variant(dataset, profile, overrides)
        created.append(path)
        typer.echo(f"  • {path}")

    typer.echo("[green]Done[/green]")


@sweep_app.command("wizard", help="Interactive wizard to create grid-search presets")
def sweep_wizard():
    """Prompt-based flow so the user can create sweeps without remembering flags."""

    import typer as _tp

    dataset = _tp.prompt("Dataset name (folder inside input/)").strip()
    profile = _tp.prompt("Profile [Flux / FluxLORA / Nude]", default="Flux").strip()

    _tp.echo("Enter parameter grid. For each line use: key=val1,val2 OR blank to finish.")
    grid_lines: list[str] = []
    while True:
        line = _tp.prompt("…", default="")
        if not line.strip():
            break
        grid_lines.append(line)

    try:
        grid = parse_grid(grid_lines)
    except ValueError as e:
        _tp.echo(f"[red]ERROR[/red] {e}")
        raise typer.Exit(code=1)

    combos = expand_grid(grid)
    _tp.echo(f"About to generate {len(combos)} presets. Continue? [y/N]")
    confirm = _tp.prompt(">", default="n").lower()
    if confirm != "y":
        _tp.echo("Aborted.")
        raise typer.Exit()

    for overrides in combos:
        path = generate_variant(dataset, profile, overrides)
        _tp.echo(f"  • {path}")

    _tp.echo("[green]Presets generated. You can now enqueue them via the Training tab in Gradio or CLI.[/green]")

# ------------------------------
# NEW: Integration enable/disable helpers
# ------------------------------

INTEGRATION_FLAGS = {
    "gsheet": "AUTO_GSHEET_ENABLE",
    "hf": "AUTO_HF_ENABLE",
    "remote": "AUTO_REMOTE_ENABLE",
}


def _integration_status(name: str) -> bool:
    """Return True if integration *name* is enabled."""
    var = INTEGRATION_FLAGS.get(name.lower())
    if not var:
        return False
    cfg = _load_int_cfg()
    return (os.getenv(var) or str(cfg.get(var, "0"))) == "1"


@int_app.command("list", help="Show available integrations and their status")
def integrations_list():
    table = Table(title="Integrations status")
    table.add_column("Integration", style="cyan")
    table.add_column("Enabled", style="green")
    for name in sorted(INTEGRATION_FLAGS):
        status = "✅" if _integration_status(name) else "❌"
        table.add_row(name, status)
    console.print(table)


@int_app.command("enable", help="Enable a specific integration")
def integrations_enable(
    name: str = typer.Argument(..., help="Integration name: gsheet | hf | remote"),
):
    name = name.lower()
    if name not in INTEGRATION_FLAGS:
        console.print(f"[red]Unknown integration '{name}'. Use 'autotrain integrations list' to see options.[/red]")
        raise typer.Exit(code=1)
    cfg = _load_int_cfg()
    cfg[INTEGRATION_FLAGS[name]] = "1"
    _save_int_cfg(cfg)
    console.print(f"[green]Integration '{name}' enabled.[/green]")


@int_app.command("disable", help="Disable a specific integration")
def integrations_disable(
    name: str = typer.Argument(..., help="Integration name: gsheet | hf | remote"),
):
    name = name.lower()
    if name not in INTEGRATION_FLAGS:
        console.print(f"[red]Unknown integration '{name}'. Use 'autotrain integrations list' to see options.[/red]")
        raise typer.Exit(code=1)
    cfg = _load_int_cfg()
    cfg[INTEGRATION_FLAGS[name]] = "0"
    _save_int_cfg(cfg)
    console.print(f"[yellow]Integration '{name}' disabled.[/yellow]")


# Interactive panel to toggle integrations quickly
@int_app.command("panel", help="Interactive panel to toggle integrations on/off")
def integrations_panel():
    """Show a table with integrations and allow toggling until user quits."""
    while True:
        integrations_list()
        choice = typer.prompt("Enter integration to toggle (blank to exit)").strip().lower()
        if not choice:
            break
        if choice not in INTEGRATION_FLAGS:
            console.print("[red]Invalid integration name.[/red]")
            continue
        # Toggle current status
        cfg = _load_int_cfg()
        var = INTEGRATION_FLAGS[choice]
        current = cfg.get(var, os.getenv(var, "0"))
        cfg[var] = "0" if str(current) == "1" else "1"
        _save_int_cfg(cfg)
        new_state = "enabled" if cfg[var] == "1" else "disabled"
        console.print(f"Integration '{choice}' is now {new_state}.")
        console.print()

from .gradio_app import _load_integrations_cfg as _load_int_cfg, _save_integrations_cfg as _save_int_cfg

# ---------------------------------------------------------------------------
# Dataset commands
# ---------------------------------------------------------------------------

@dataset_app.command("create", help="Create folders in 'input/' for the given names.")
def dataset_create(names: str = typer.Option(..., help="Names separated by comma, e.g. 'alex,maria'")) -> None:
    names_list = [n.strip() for n in names.split(",") if n.strip()]
    if not names_list:
        typer.echo("You must provide at least one name")
        raise typer.Exit(code=1)
    create_input_folders(names_list)


@dataset_app.command("build-output", help="Create structure under 'output/' and copy images.")
def dataset_build_output(min_images: int = typer.Option(0, help="Minimum number of images per folder")) -> None:
    try:
        populate_output_structure(min_images=min_images)
    except Exception as e:
        console.print(f"[red]ERROR:[/red] {e}")
        raise typer.Exit(code=1)


@dataset_app.command("clean", help="Remove workspace folders")
def dataset_clean(
    input: bool = typer.Option(True, "--input/--no-input", help="Delete 'input' folder"),
    output: bool = typer.Option(True, "--output/--no-output", help="Delete 'output' folder"),
    batchconfig: bool = typer.Option(True, "--batchconfig/--no-batchconfig", help="Delete 'BatchConfig' folder"),
):
    clean_workspace(delete_input=input, delete_output=output, delete_batchconfig=batchconfig)


@dataset_app.command("create-prompts", help="Create sample_prompts.txt files for all datasets in output/")
def dataset_create_prompts() -> None:
    """Create sample_prompts.txt files for all existing datasets in output/ directory."""
    try:
        count = create_sample_prompts()
        if count > 0:
            console.print(f"[green]✓[/green] Created/updated {count} sample_prompts.txt files")
        else:
            console.print("[yellow]No datasets found in output/ directory[/yellow]")
    except Exception as e:
        console.print(f"[red]ERROR:[/red] {e}")
        raise typer.Exit(code=1)


@dataset_app.command("create-prompts-single", help="Create sample_prompts.txt file for a specific dataset")
def dataset_create_prompts_single(
    dataset: str = typer.Argument(..., help="Dataset name (folder in output/)")
) -> None:
    """Create sample_prompts.txt file for a specific dataset."""
    try:
        success = create_sample_prompts_for_dataset(dataset)
        if success:
            console.print(f"[green]✓[/green] Created sample_prompts.txt for dataset '{dataset}'")
        else:
            console.print(f"[red]✗[/red] Failed to create sample_prompts.txt for dataset '{dataset}'")
            console.print("[yellow]Make sure the dataset exists in output/ directory[/yellow]")
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]ERROR:[/red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Config commands
# ---------------------------------------------------------------------------

@config_app.command("refresh", help="Regenerate TOML presets by running combined_script.py")
def config_refresh() -> None:
    generate_presets()
    console.print("[green]Presets generated successfully[/green]")


@config_app.command("show", help="Display a TOML file as a table")
def config_show(file: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    cfg = load_config(file)
    table = Table(title=str(file))
    table.add_column("Key")
    table.add_column("Value")
    for k, v in cfg.items():
        table.add_row(str(k), str(v))
    console.print(table)


@config_app.command("set", help="Update key=value pairs inside a TOML file")
def config_set(
    file: Path = typer.Argument(..., exists=True, readable=True, writable=True),
    kv: List[str] = typer.Option(..., help="List of key=value pairs"),
):
    updates = {}
    for pair in kv:
        if "=" not in pair:
            typer.echo(f"Incorrect format: '{pair}'. Expected key=value")
            raise typer.Exit(1)
        key, value = pair.split("=", 1)
        updates[key.strip()] = _coerce_value(value.strip())
    update_config(file, updates)
    console.print(f"[green]Updated {file}[/green]")


def _coerce_value(value: str):
    """Try to convert strings to int, float or bool when possible"""
    lowers = value.lower()
    if lowers in {"true", "false"}:
        return lowers == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# Train commands
# ---------------------------------------------------------------------------

@train_app.command("start", help="Start a training using a TOML file")
def train_start(
    profile: str = typer.Option(..., help="Profile type: Flux / FluxLORA / Nude"),
    file: Path = typer.Argument(..., exists=True, readable=True),
    now: bool = typer.Option(False, "--now", help="Run immediately (blocking)"),
    gpu: str = typer.Option(None, "--gpu", help="GPU IDs, e.g. '0,1'"),
):
    """Launch a training.  By default the job is enqueued in the JobManager."""

    if now:
        rc = run_training(file, profile, stream=True, gpu_ids=gpu)

        # ----- Manual call to integrations so immediate runs are logged -----
        try:
            import toml, json
            from .integrations import handle_job_complete  # type: ignore
            from .job_manager import Job, JobStatus

            cfg = toml.load(file)
            run_dir = Path(cfg.get("output_dir", "output"))

            job = Job(dataset=file.stem, profile=profile, toml_path=file, run_dir=run_dir)
            job.status = JobStatus.DONE if rc == 0 else JobStatus.FAILED
            job.total_steps = cfg.get("max_train_steps", 0)
            # minimal progress string
            job.progress_str = "100%" if rc == 0 else "error"

            handle_job_complete(job)  # type: ignore[arg-type]
        except Exception:
            pass  # silent: not critical for training path

        sys.exit(rc)

    dataset_name = file.stem
    run_dir = compute_run_dir(dataset_name, profile)
    console.print(f"Output dir → [cyan]{run_dir}[/cyan]")
    job = Job(dataset_name, profile, file, run_dir, gpu_ids=gpu)
    JOB_MANAGER.enqueue(job)
    console.print(f"[green]Job {job.id} enqueued[/green]")


# ---------------------------------------------------------------------------
# Enhanced Train commands
# ---------------------------------------------------------------------------

@train_app.command("info", help="Show dataset information and training estimation")
def train_info(
    datasets: str = typer.Argument(..., help="Dataset name(s), comma-separated"),
    profile: str = typer.Option("Flux", help="Profile: Flux / FluxLORA / Nude"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
):
    """Display detailed information about datasets and training estimation."""
    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
    
    for dataset_name in dataset_list:
        console.print()
        
        # Dataset Information
        info_table = _create_dataset_info_table(dataset_name, profile)
        console.print(info_table)
        
        console.print()
        
        # Training Estimation
        estimation_table = _create_training_estimation_table(dataset_name, profile)
        console.print(estimation_table)
        
        if len(dataset_list) > 1 and dataset_name != dataset_list[-1]:
            console.print("\n" + "─" * 80 + "\n")


@train_app.command("queue", help="Manage training queue")
def train_queue(
    action: str = typer.Argument("list", help="Action: list, cancel, clear, status"),
    job_id: Optional[str] = typer.Argument(None, help="Job ID for cancel/status actions"),
    force: bool = typer.Option(False, "--force", "-f", help="Force clear without confirmation"),
):
    """Manage the training queue - list, cancel, clear, or check status."""
    
    if action == "list":
        queue_table = _create_queue_table()
        console.print(queue_table)
        
    elif action == "cancel":
        if not job_id:
            console.print("[red]Error:[/red] Job ID required for cancel action")
            raise typer.Exit(code=1)
        
        job = None
        for j in JOB_MANAGER.list_jobs():
            if j.id == job_id:
                job = j
                break
        
        if not job:
            console.print(f"[red]Error:[/red] Job {job_id} not found")
            raise typer.Exit(code=1)
        
        JOB_MANAGER.cancel(job_id)
        console.print(f"[yellow]Job {job_id} canceled[/yellow]")
        
    elif action == "clear":
        if not force:
            confirm = typer.confirm("Are you sure you want to clear the entire queue?")
            if not confirm:
                console.print("Operation canceled")
                raise typer.Exit()
        
        JOB_MANAGER.clear_queue()
        console.print("[yellow]Queue cleared[/yellow]")
        
    elif action == "status":
        if not job_id:
            console.print("[red]Error:[/red] Job ID required for status action")
            raise typer.Exit(code=1)
        
        job = None
        for j in JOB_MANAGER.list_jobs():
            if j.id == job_id:
                job = j
                break
        
        if not job:
            console.print(f"[red]Error:[/red] Job {job_id} not found")
            raise typer.Exit(code=1)
        
        # Create detailed status panel
        status_panel = _create_status_panel(job)
        console.print(status_panel)
        
        if job.status == JobStatus.RUNNING:
            metrics_panel = _create_metrics_panel(job)
            console.print(metrics_panel)
    
    else:
        console.print(f"[red]Error:[/red] Unknown action '{action}'. Use: list, cancel, clear, status")
        raise typer.Exit(code=1)


@train_app.command("monitor", help="Monitor active training with real-time metrics")  
def train_monitor(
    job_id: Optional[str] = typer.Option(None, "--job", help="Monitor specific job ID"),
    refresh: int = typer.Option(3, "--refresh", "-r", help="Refresh interval in seconds"),
    compact: bool = typer.Option(False, "--compact", "-c", help="Compact display mode"),
):
    """Monitor active training with real-time metrics and logs."""
    
    def get_current_job():
        if job_id:
            for job in JOB_MANAGER.list_jobs():
                if job.id == job_id:
                    return job
            return None
        else:
            return JOB_MANAGER.get_current_job()
    
    def create_monitoring_layout(job: Job):
        """Create the monitoring layout."""
        if compact:
            # Compact mode - single column
            layout = Layout()
            layout.split_column(
                Layout(_create_status_panel(job), size=6),
                Layout(_create_metrics_panel(job), size=12),
                Layout(_create_log_panel(job), size=8),
            )
        else:
            # Full mode - two columns
            layout = Layout()
            layout.split_column(
                Layout(Panel(Text(f"🔥 Training Monitor - Job: {job.id}\nDataset: {job.dataset} | Profile: {job.profile}", 
                                 justify="center"), title="AutoTrain Monitor", border_style="green"), size=3),
                Layout().split_row(
                    Layout().split_column(
                        Layout(_create_metrics_panel(job), size=15),
                        Layout(_create_status_panel(job), size=6)
                    ),
                    Layout(_create_log_panel(job))
                )
            )
        return layout
    
    # Initial check
    job = get_current_job()
    if not job:
        if job_id:
            console.print(f"[red]Error:[/red] Job {job_id} not found")
        else:
            console.print("[yellow]No active training job found[/yellow]")
        raise typer.Exit(code=1)
    
    console.print(f"[green]Monitoring job {job.id} - Press Ctrl+C to exit[/green]")
    console.print("Commands: [bold]q[/bold] = quit, [bold]r[/bold] = manual refresh, [bold]c[/bold] = cancel training")
    
    try:
        with Live(create_monitoring_layout(job), refresh_per_second=1/refresh, screen=True) as live:
            while True:
                time.sleep(refresh)
                
                # Get updated job
                current_job = get_current_job()
                if not current_job:
                    console.print("\n[yellow]Job completed or no longer active[/yellow]")
                    break
                
                if current_job.status in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED]:
                    live.update(create_monitoring_layout(current_job))
                    console.print(f"\n[yellow]Job {current_job.status.lower()}[/yellow]")
                    break
                
                # Update display
                live.update(create_monitoring_layout(current_job))
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


@train_app.command("interactive", help="Interactive training setup wizard")
def train_interactive():
    """Interactive wizard for setting up training with prompts and selections."""
    
    console.print(Panel(Text("🎯 Interactive Training Setup", justify="center"), 
                       title="AutoTrain Wizard", border_style="green"))
    
    # Training Mode Selection
    console.print("\n[bold blue]Training Mode:[/bold blue]")
    mode = typer.prompt("Select mode", type=click.Choice(['single', 'batch']), default='single')
    
    # Profile Selection  
    console.print("\n[bold blue]Profile Selection:[/bold blue]")
    profile = typer.prompt("Select profile", type=click.Choice(['Flux', 'FluxLORA', 'Nude']), default='Flux')
    
    # Dataset Selection
    from .gradio_app import _list_dataset_raw
    available_datasets = _list_dataset_raw()
    
    if not available_datasets:
        console.print("[red]No datasets found in output/ directory[/red]")
        console.print("Please run: [cyan]autotrain dataset build-output[/cyan]")
        raise typer.Exit(code=1)
    
    console.print("\n[bold blue]Available Datasets:[/bold blue]")
    for i, dataset in enumerate(available_datasets, 1):
        dataset_info = _get_dataset_info(dataset)
        size = dataset_info.get('total_size', 'N/A')
        images = dataset_info.get('num_images', 0)
        console.print(f"  {i}. {dataset} ({images} images, {size})")
    
    if mode == 'single':
        console.print()
        dataset_idx = typer.prompt("Select dataset number", type=int) - 1
        if dataset_idx < 0 or dataset_idx >= len(available_datasets):
            console.print("[red]Invalid selection[/red]")
            raise typer.Exit(code=1)
        selected_datasets = [available_datasets[dataset_idx]]
    else:
        console.print()
        console.print("Enter dataset numbers separated by commas (e.g., 1,3,5):")
        indices = typer.prompt("Dataset numbers").split(',')
        selected_datasets = []
        for idx_str in indices:
            try:
                idx = int(idx_str.strip()) - 1
                if 0 <= idx < len(available_datasets):
                    selected_datasets.append(available_datasets[idx])
            except ValueError:
                console.print(f"[yellow]Invalid number: {idx_str}[/yellow]")
        
        if not selected_datasets:
            console.print("[red]No valid datasets selected[/red]")
            raise typer.Exit(code=1)
    
    # Advanced Options
    console.print("\n[bold blue]Advanced Options:[/bold blue]")
    gpu_ids = typer.prompt("GPU IDs (e.g., '0,1' or leave empty)", default="", show_default=False)
    modify_params = typer.confirm("Modify training parameters?", default=False)
    
    # Configuration Summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Configuration Summary:[/bold green]")
    console.print(f"  Mode:      {mode}")
    console.print(f"  Profile:   {profile}") 
    console.print(f"  Datasets:  {', '.join(selected_datasets)}")
    if gpu_ids:
        console.print(f"  GPU IDs:   {gpu_ids}")
    
    # Show time estimation for first dataset
    if selected_datasets:
        estimation = _estimate_training_time(selected_datasets[0], profile)
        if "estimated_time" in estimation:
            console.print(f"  Time Est:  {estimation['estimated_time']}")
    
    console.print("=" * 60)
    
    # Confirmation
    if not typer.confirm("\nStart training now?", default=True):
        console.print("Training canceled")
        raise typer.Exit()
    
    # Start training
    for dataset_name in selected_datasets:
        # Build TOML path
        profile_dir = "Flux" if profile == "Flux" else ("FluxLORA" if profile == "FluxLORA" else "Nude")
        cfg_path = BATCH_CONFIG_DIR / profile_dir / f"{dataset_name}.toml"
        
        if not cfg_path.exists():
            console.print(f"[red]Error:[/red] Config file not found: {cfg_path}")
            console.print("Please run: [cyan]autotrain config refresh[/cyan]")
            continue
        
        # Create and enqueue job
        run_dir = compute_run_dir(dataset_name, profile)
        job = Job(dataset_name, profile, cfg_path, run_dir, gpu_ids=gpu_ids or None)
        JOB_MANAGER.enqueue(job)
        
        console.print(f"[green]✓[/green] Job {job.id} queued for dataset {dataset_name}")
    
    console.print(f"\n[green]Successfully queued {len(selected_datasets)} job(s)[/green]")
    console.print("Use [cyan]autotrain train monitor[/cyan] to watch progress")


@train_app.command("batch", help="Batch training operations")
def train_batch(
    datasets: str = typer.Option(..., "--datasets", help="Comma-separated dataset names"),
    profile: str = typer.Option("Flux", "--profile", help="Profile: Flux / FluxLORA / Nude"),
    gpu_ids: Optional[str] = typer.Option(None, "--gpu", help="GPU IDs, e.g. '0,1'"),
    priority: str = typer.Option("normal", "--priority", help="Priority: low, normal, high"),
):
    """Start batch training for multiple datasets in sequence."""
    
    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
    
    if not dataset_list:
        console.print("[red]Error:[/red] No datasets specified")
        raise typer.Exit(code=1)
    
    console.print(f"[blue]Preparing batch training for {len(dataset_list)} datasets...[/blue]")
    
    # Validate all datasets first
    valid_datasets = []
    for dataset_name in dataset_list:
        dataset_info = _get_dataset_info(dataset_name)
        if not dataset_info.get("exists", False):
            console.print(f"[yellow]Warning:[/yellow] Dataset '{dataset_name}' not found, skipping")
            continue
        valid_datasets.append(dataset_name)
    
    if not valid_datasets:
        console.print("[red]Error:[/red] No valid datasets found")
        raise typer.Exit(code=1)
    
    console.print(f"[green]Found {len(valid_datasets)} valid datasets[/green]")
    
    # Show summary
    total_time = 0
    for dataset_name in valid_datasets:
        estimation = _estimate_training_time(dataset_name, profile)
        console.print(f"  • {dataset_name}: {estimation.get('estimated_time', 'N/A')}")
        
        # Try to parse time for total calculation (simple parsing)
        time_str = estimation.get('estimated_time', '')
        if 'h' in time_str and 'm' in time_str:
            try:
                parts = time_str.replace('h', '').replace('m', '').split()
                if len(parts) >= 2:
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    total_time += hours * 60 + minutes
            except:
                pass
    
    if total_time > 0:
        total_hours = int(total_time // 60)
        total_minutes = int(total_time % 60)
        console.print(f"\n[bold]Estimated total time: {total_hours}h {total_minutes}m[/bold]")
    
    # Confirmation
    if not typer.confirm(f"\nQueue {len(valid_datasets)} training jobs?", default=True):
        console.print("Batch training canceled")
        raise typer.Exit()
    
    # Queue all jobs
    queued_jobs = []
    for dataset_name in valid_datasets:
        profile_dir = "Flux" if profile == "Flux" else ("FluxLORA" if profile == "FluxLORA" else "Nude")
        cfg_path = BATCH_CONFIG_DIR / profile_dir / f"{dataset_name}.toml"
        
        if not cfg_path.exists():
            console.print(f"[yellow]Warning:[/yellow] Config not found for {dataset_name}, skipping")
            continue
        
        run_dir = compute_run_dir(dataset_name, profile)
        job = Job(dataset_name, profile, cfg_path, run_dir, gpu_ids=gpu_ids)
        JOB_MANAGER.enqueue(job)
        queued_jobs.append(job)
        
        console.print(f"[green]✓[/green] Job {job.id} queued: {dataset_name}")
    
    console.print(f"\n[green]Successfully queued {len(queued_jobs)} jobs[/green]")
    console.print("Use [cyan]autotrain train queue list[/cyan] to see the queue")
    console.print("Use [cyan]autotrain train monitor[/cyan] to watch progress")


# ---------------------------------------------------------------------------
# Web commands
# ---------------------------------------------------------------------------

@a_pp.command("serve", help="Start Gradio server")
def web_serve(
    share: bool = typer.Option(False, "--share", help="Enable public share tunnel")
):
    launch_web(share=share)


# ---------------------------------------------------------------------------
# Integrations – Google Sheets
# ---------------------------------------------------------------------------

@int_app.command("gsheet", help="Set Google Sheets credentials and sheet info")
def integrations_gsheet(
    cred: Path = typer.Option(..., exists=True, readable=True, help="Path to service-account JSON file"),
    sheet_id: str = typer.Option(..., help="Spreadsheet ID (long hash in URL)"),
    tab: Optional[str] = typer.Option(None, help="Worksheet name inside the spreadsheet"),
):
    """Store Google Sheets configuration in integrations file and current env vars."""
    cfg = _load_int_cfg()
    cfg["AUTO_GSHEET_CRED"] = str(cred)
    cfg["AUTO_GSHEET_ID"] = sheet_id.strip()
    cfg["AUTO_GSHEET_TAB"] = tab.strip() if tab else None
    _save_int_cfg(cfg)
    console.print("[green]Google Sheets configuration saved[/green]")

# Convenience command to show current integration settings

@int_app.command("show", help="Show current integration config")
def integrations_show():
    cfg = _load_int_cfg()
    if not cfg:
        console.print("[yellow]No integration config found[/yellow]")
        raise typer.Exit()
    table = Table(title="Integrations config")
    table.add_column("Key")
    table.add_column("Value")
    for k, v in cfg.items():
        table.add_row(k, str(v))
    console.print(table)

# ---------------------------------------------------------------------------
# Registry commands implementation
# ---------------------------------------------------------------------------

@registry_app.command("rebuild", help="Rebuild runs.json scanning output/")
def registry_rebuild():
    from .run_registry import rebuild as _rr_rebuild

    def _progress(done, total):
        console.print(f"[cyan]Scanning[/cyan] {done}/{total} runs", end="\r")

    total = _rr_rebuild(_progress)
    console.print(f"\n[green]Rebuilt registry with {total} runs.[/green]")

# ---------------------------------------------------------------------------
# New: caption subcommands
# ---------------------------------------------------------------------------

@caption_app.command("run", help="Generate image captions using Florence-2")
def caption_run(
    names: str = typer.Argument(..., help="Dataset names separated by comma, e.g. 'b09g13,dl4r0s4'"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite caption .txt files if they already exist"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Maximum tokens to generate per caption"),
    device: str | None = typer.Option(None, "--device", help="Force device: 'cuda', 'cpu', 'cuda:1', …"),
):
    """CLI wrapper around :pyfunc:`autotrain_sdk.captioner.caption_dataset`."""

    from .captioner import caption_dataset

    datasets = [n.strip() for n in names.split(",") if n.strip()]
    if not datasets:
        console.print("[red]You must provide at least one dataset name.[/red]")
        raise typer.Exit(code=1)

    total = 0
    for ds in datasets:
        try:
            created = caption_dataset(ds, device=device, max_new_tokens=max_tokens, overwrite=overwrite)
            console.print(f"[green]{ds}[/green] → {created} caption(s) generated.")
            total += created
        except FileNotFoundError as e:
            console.print(f"[yellow]WARN[/yellow] {e}")

    console.print(f"[bold]{total} caption(s) generated in total[/bold]")

# ---------------------------------------------------------------------------
# Enhanced train start command
# ---------------------------------------------------------------------------

@train_app.command("status", help="Show current training status")
def train_status(
    all_jobs: bool = typer.Option(False, "--all", "-a", help="Show all jobs (not just active)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
):
    """Show current training status and active jobs."""
    
    # Get jobs
    jobs = JOB_MANAGER.list_jobs()
    
    if not all_jobs:
        # Filter only active jobs
        active_jobs = [job for job in jobs if job.status in [JobStatus.RUNNING, JobStatus.PENDING]]
        jobs = active_jobs
    
    if not jobs:
        console.print("[yellow]No active training jobs[/yellow]")
        return
    
    # Show queue table
    queue_table = _create_queue_table() if not all_jobs else _create_queue_table()
    console.print(queue_table)
    
    # Show detailed status for running jobs
    if verbose:
        running_jobs = [job for job in jobs if job.status == JobStatus.RUNNING]
        for job in running_jobs:
            console.print()
            status_panel = _create_status_panel(job)
            console.print(status_panel)
            
            metrics_panel = _create_metrics_panel(job)
            console.print(metrics_panel)


@train_app.command("logs", help="Show training logs")
def train_logs(
    job_id: Optional[str] = typer.Option(None, "--job", help="Specific job ID"),
    tail: int = typer.Option(50, "--tail", "-n", help="Show last N lines"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """Show training logs for active or specified job."""
    
    job = None
    if job_id:
        for j in JOB_MANAGER.list_jobs():
            if j.id == job_id:
                job = j
                break
        if not job:
            console.print(f"[red]Error:[/red] Job {job_id} not found")
            raise typer.Exit(code=1)
    else:
        job = JOB_MANAGER.get_current_job()
        if not job:
            console.print("[yellow]No active training job found[/yellow]")
            raise typer.Exit(code=1)
    
    console.print(f"[blue]Logs for job {job.id} ({job.dataset})[/blue]")
    
    if follow:
        console.print("[dim]Press Ctrl+C to stop following[/dim]")
        try:
            while True:
                log, _ = JOB_MANAGER.get_live_output(job.id)
                if log:
                    lines = log.split('\n')
                    recent_lines = lines[-tail:] if len(lines) > tail else lines
                    console.print('\n'.join(recent_lines))
                
                time.sleep(2)
                
                # Check if job is still running
                updated_job = None
                for j in JOB_MANAGER.list_jobs():
                    if j.id == job.id:
                        updated_job = j
                        break
                
                if not updated_job or updated_job.status in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED]:
                    console.print(f"[yellow]Job {updated_job.status.lower() if updated_job else 'completed'}[/yellow]")
                    break
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped following logs[/yellow]")
    else:
        log, _ = JOB_MANAGER.get_live_output(job.id)
        if log:
            lines = log.split('\n')
            recent_lines = lines[-tail:] if len(lines) > tail else lines
            console.print('\n'.join(recent_lines))
        else:
            console.print("[yellow]No logs available yet[/yellow]")


@train_app.command("stop", help="Stop active training")
def train_stop(
    job_id: Optional[str] = typer.Option(None, "--job", help="Specific job ID to stop"),
    force: bool = typer.Option(False, "--force", "-f", help="Force stop without confirmation"),
):
    """Stop active training job."""
    
    job = None
    if job_id:
        for j in JOB_MANAGER.list_jobs():
            if j.id == job_id:
                job = j
                break
        if not job:
            console.print(f"[red]Error:[/red] Job {job_id} not found")
            raise typer.Exit(code=1)
    else:
        job = JOB_MANAGER.get_current_job()
        if not job:
            console.print("[yellow]No active training job to stop[/yellow]")
            raise typer.Exit(code=1)
    
    if job.status != JobStatus.RUNNING:
        console.print(f"[yellow]Job {job.id} is not running (status: {job.status})[/yellow]")
        return
    
    if not force:
        console.print(f"[yellow]Job Details:[/yellow]")
        console.print(f"  ID: {job.id}")
        console.print(f"  Dataset: {job.dataset}")
        console.print(f"  Profile: {job.profile}")
        console.print(f"  Progress: {job.percent:.1f}%")
        
        if not typer.confirm(f"Are you sure you want to stop job {job.id}?"):
            console.print("Operation canceled")
            return
    
    JOB_MANAGER.cancel(job.id)
    console.print(f"[green]Job {job.id} stopped[/green]")


@train_app.command("history", help="Show training history")
def train_history(
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of jobs to show"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status: done, failed, canceled"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Filter by dataset name"),
):
    """Show training history with filters."""
    
    jobs = JOB_MANAGER.list_jobs()
    
    # Apply filters
    if status:
        status_filter = getattr(JobStatus, status.upper(), None)
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
    
    if dataset:
        jobs = [job for job in jobs if dataset.lower() in job.dataset.lower()]
    
    # Sort by most recent first (assuming job.id contains timestamp)
    jobs = sorted(jobs, key=lambda x: x.id, reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    if not jobs:
        console.print("[yellow]No jobs found matching criteria[/yellow]")
        return
    
    # Create history table
    table = Table(title="Training History")
    table.add_column("Job ID", style="cyan")
    table.add_column("Dataset", style="blue")
    table.add_column("Profile", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Progress", style="green")
    table.add_column("Duration", style="yellow")
    
    for job in jobs:
        # Calculate duration if available
        duration = "N/A"
        if hasattr(job, 'start_time') and hasattr(job, 'end_time'):
            if job.start_time and job.end_time:
                duration_seconds = job.end_time - job.start_time
                duration = _format_duration(duration_seconds)
        elif hasattr(job, 'elapsed') and job.elapsed:
            duration = job.elapsed
        
        progress_text = f"{job.percent:.1f}%" if job.percent > 0 else "N/A"
        
        table.add_row(
            job.id,
            job.dataset,
            job.profile,
            _format_job_status(job.status),
            progress_text,
            duration
        )
    
    console.print(table)


@train_app.command("clean", help="Clean completed/failed jobs from queue")
def train_clean(
    status: str = typer.Option("done", "--status", help="Status to clean: done, failed, canceled, all"),
    force: bool = typer.Option(False, "--force", "-f", help="Force clean without confirmation"),
):
    """Clean completed or failed jobs from the queue."""
    
    jobs = JOB_MANAGER.list_jobs()
    
    # Determine which jobs to clean
    if status == "all":
        jobs_to_clean = [job for job in jobs if job.status in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED]]
    else:
        status_filter = getattr(JobStatus, status.upper(), None)
        if not status_filter:
            console.print(f"[red]Error:[/red] Invalid status '{status}'. Use: done, failed, canceled, all")
            raise typer.Exit(code=1)
        jobs_to_clean = [job for job in jobs if job.status == status_filter]
    
    if not jobs_to_clean:
        console.print(f"[yellow]No jobs with status '{status}' found[/yellow]")
        return
    
    console.print(f"[yellow]Jobs to clean ({len(jobs_to_clean)}):[/yellow]")
    for job in jobs_to_clean:
        console.print(f"  • {job.id} ({job.dataset}) - {job.status}")
    
    if not force:
        if not typer.confirm(f"Remove {len(jobs_to_clean)} job(s) from queue?"):
            console.print("Operation canceled")
            return
    
    # Clean jobs using JobManager methods
    removed_count = 0
    for job in jobs_to_clean:
        try:
            if JOB_MANAGER.remove_job(job.id):
                removed_count += 1
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not remove job {job.id}: {e}")
    
    console.print(f"[green]Cleaned {removed_count} job(s) from queue[/green]")


# ---------------------------------------------------------------------------
# Pipeline commands implementation
# ---------------------------------------------------------------------------

@pipeline_app.command("run", help="Run complete training pipeline from dataset to monitoring")
def pipeline_run(
    dataset_path: str = typer.Option(..., "--dataset-path", help="Path to external dataset directory"),
    profile: str = typer.Option(..., "--profile", help="Training profile: Flux / FluxLORA / Nude"),
    dataset_name: Optional[str] = typer.Option(None, "--dataset-name", help="Custom dataset name (default: folder name)"),
    monitor: bool = typer.Option(False, "--monitor", help="Start live monitoring after training begins"),
    min_images: int = typer.Option(1, "--min-images", help="Minimum number of images required"),
    gpu_ids: Optional[str] = typer.Option(None, "--gpu", help="GPU IDs to use, e.g. '0,1'"),
    skip_copy: bool = typer.Option(False, "--skip-copy", help="Skip copying dataset if it already exists"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing datasets"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing"),
    immediate: bool = typer.Option(False, "--immediate", help="Run training immediately instead of queueing"),
):
    """
    Execute the complete training pipeline from external dataset to live monitoring.
    
    This command automates the entire process:
    1. Copy dataset from external path to input/
    2. Prepare output structure
    3. Generate TOML configuration
    4. Start training job
    5. Monitor progress (if requested)
    
    Example:
        autotrain pipeline run --dataset-path /home/user/dataset --profile Flux --monitor
    """
    from .pipeline import run_pipeline
    
    success = run_pipeline(
        dataset_path=dataset_path,
        profile=profile,
        dataset_name=dataset_name,
        monitor=monitor,
        min_images=min_images,
        gpu_ids=gpu_ids,
        skip_copy=skip_copy,
        force=force,
        dry_run=dry_run,
        immediate=immediate,
    )
    
    if not success:
        console.print("[red]Pipeline execution failed[/red]")
        raise typer.Exit(code=1)


@pipeline_app.command("prepare", help="Prepare dataset and configuration without training")
def pipeline_prepare(
    dataset_path: str = typer.Option(..., "--dataset-path", help="Path to external dataset directory"),
    profile: str = typer.Option(..., "--profile", help="Training profile: Flux / FluxLORA / Nude"),
    dataset_name: Optional[str] = typer.Option(None, "--dataset-name", help="Custom dataset name (default: folder name)"),
    min_images: int = typer.Option(1, "--min-images", help="Minimum number of images required"),
    skip_copy: bool = typer.Option(False, "--skip-copy", help="Skip copying dataset if it already exists"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing datasets"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing"),
):
    """
    Prepare dataset and configuration without starting training.
    
    This is useful for batch preparation or when you want to review
    configurations before starting training.
    
    Example:
        autotrain pipeline prepare --dataset-path /home/user/dataset --profile Flux
    """
    from .pipeline import prepare_pipeline
    
    success = prepare_pipeline(
        dataset_path=dataset_path,
        profile=profile,
        dataset_name=dataset_name,
        min_images=min_images,
        skip_copy=skip_copy,
        force=force,
        dry_run=dry_run,
    )
    
    if not success:
        console.print("[red]Pipeline preparation failed[/red]")
        raise typer.Exit(code=1)


@pipeline_app.command("status", help="Show pipeline status and recent operations")
def pipeline_status():
    """Show status of recent pipeline operations and current training jobs."""
    
    # Show current training jobs
    console.print("[bold blue]Current Training Jobs:[/bold blue]")
    queue_table = _create_queue_table()
    console.print(queue_table)
    
    # Show available datasets
    console.print("\n[bold blue]Available Datasets:[/bold blue]")
    from .gradio_app import _list_dataset_raw
    datasets = _list_dataset_raw()
    
    if datasets:
        dataset_table = Table()
        dataset_table.add_column("Dataset", style="cyan")
        dataset_table.add_column("Images", style="green")
        dataset_table.add_column("Size", style="yellow")
        
        for dataset in datasets[:10]:  # Show first 10
            from .gradio_app import _get_dataset_info
            info = _get_dataset_info(dataset)
            dataset_table.add_row(
                dataset,
                str(info.get('num_images', 'N/A')),
                info.get('total_size', 'N/A')
            )
        
        console.print(dataset_table)
        
        if len(datasets) > 10:
            console.print(f"[dim]... and {len(datasets) - 10} more datasets[/dim]")
    else:
        console.print("[yellow]No datasets found in output/ directory[/yellow]")


@pipeline_app.command("batch", help="Run pipeline for multiple datasets in batch")
def pipeline_batch(
    datasets_dir: Optional[str] = typer.Option(None, "--datasets-dir", help="Directory containing multiple dataset folders"),
    dataset_paths: Optional[str] = typer.Option(None, "--dataset-paths", help="Comma-separated list of dataset paths"),
    profile: str = typer.Option("FluxLORA", "--profile", help="Training profile: Flux / FluxLORA / Nude"),
    monitor: bool = typer.Option(False, "--monitor", help="Start live monitoring after all jobs are enqueued"),
    min_images: int = typer.Option(1, "--min-images", help="Minimum number of images required per dataset"),
    gpu_ids: Optional[str] = typer.Option(None, "--gpu", help="GPU IDs to use, e.g. '0,1'"),
    skip_copy: bool = typer.Option(False, "--skip-copy", help="Skip copying datasets if they already exist"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing datasets"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing"),
    max_concurrent: int = typer.Option(1, "--max-concurrent", help="Maximum concurrent jobs (currently only 1 supported)"),
):
    """
    Execute the pipeline for multiple datasets in batch.
    
    This command automates the entire process for multiple datasets:
    1. Auto-discover or parse dataset paths
    2. Prepare each dataset (copy, structure, config)
    3. Enqueue all training jobs
    4. Monitor progress (if requested)
    
    Examples:
        # Auto-discover datasets in a directory
        autotrain pipeline batch --datasets-dir /path/to/datasets --profile FluxLORA --monitor
        
        # Process specific datasets
        autotrain pipeline batch --dataset-paths "/path/to/dataset1,/path/to/dataset2" --profile Flux
        
        # Dry run to see what would be processed
        autotrain pipeline batch --datasets-dir /path/to/datasets --profile FluxLORA --dry-run
    """
    from .pipeline import run_batch_pipeline
    
    try:
        success = run_batch_pipeline(
            datasets_dir=datasets_dir,
            dataset_paths=dataset_paths,
            profile=profile,
            monitor=monitor,
            min_images=min_images,
            gpu_ids=gpu_ids,
            skip_copy=skip_copy,
            force=force,
            dry_run=dry_run,
            max_concurrent=max_concurrent,
        )
        
        if success:
            console.print("\n[green]✅ Batch pipeline completed successfully![/green]")
        else:
            console.print("\n[red]❌ Batch pipeline failed![/red]")
            raise typer.Exit(code=1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Batch pipeline interrupted by user[/yellow]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[red]❌ Batch pipeline error: {e}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app() 
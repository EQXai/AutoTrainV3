"""
Automated Training Pipeline for AutoTrainV2

This module provides a unified command-line interface for automating the complete
training pipeline from external dataset to live monitoring.

Usage:
    autotrain pipeline run --dataset-path /path/to/dataset --profile Flux --monitor
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

from .dataset import (
    populate_output_structure_single,
    create_sample_prompts_for_dataset,
)
from .configurator import generate_presets_for_dataset
from .trainer import run_training
from .job_manager import JOB_MANAGER, Job, JobStatus
from .paths import INPUT_DIR, OUTPUT_DIR, BATCH_CONFIG_DIR, compute_run_dir

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
VALID_PROFILES = ['Flux', 'FluxLORA', 'Nude']


class PipelineValidator:
    """Validates pipeline inputs and prerequisites."""
    
    @staticmethod
    def validate_dataset_path(dataset_path: Path) -> Tuple[bool, str]:
        """Validate that the dataset path exists and contains valid data."""
        if not dataset_path.exists():
            return False, f"Dataset path does not exist: {dataset_path}"
        
        if not dataset_path.is_dir():
            return False, f"Dataset path is not a directory: {dataset_path}"
        
        # Check for image files
        image_files = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(dataset_path.glob(f"*{ext}"))
            image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            return False, f"No image files found in dataset path: {dataset_path}"
        
        return True, f"Found {len(image_files)} image files"
    
    @staticmethod
    def validate_profile(profile: str) -> Tuple[bool, str]:
        """Validate that the profile is supported."""
        if profile not in VALID_PROFILES:
            return False, f"Invalid profile '{profile}'. Valid profiles: {', '.join(VALID_PROFILES)}"
        return True, f"Profile '{profile}' is valid"
    
    @staticmethod
    def validate_dataset_name(dataset_name: str) -> Tuple[bool, str]:
        """Validate dataset name format."""
        if not dataset_name:
            return False, "Dataset name cannot be empty"
        
        if not dataset_name.replace('_', '').replace('-', '').isalnum():
            return False, "Dataset name can only contain letters, numbers, hyphens, and underscores"
        
        return True, f"Dataset name '{dataset_name}' is valid"
    
    @staticmethod
    def check_disk_space(required_gb: float = 5.0) -> Tuple[bool, str]:
        """Check if there's enough disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb < required_gb:
                return False, f"Insufficient disk space. Required: {required_gb:.1f}GB, Available: {free_gb:.1f}GB"
            
            return True, f"Sufficient disk space available: {free_gb:.1f}GB"
        except Exception as e:
            return False, f"Could not check disk space: {e}"


class PipelineExecutor:
    """Executes the training pipeline steps."""
    
    def __init__(self, dataset_path: Path, profile: str, dataset_name: Optional[str] = None):
        self.dataset_path = dataset_path
        self.profile = profile
        self.dataset_name = dataset_name or dataset_path.name
        self.input_dataset_path = INPUT_DIR / self.dataset_name
        self.output_dataset_path = OUTPUT_DIR / self.dataset_name
        self.config_path = BATCH_CONFIG_DIR / self.profile / f"{self.dataset_name}.toml"
        
    def copy_dataset(self, force: bool = False, skip_copy: bool = False) -> bool:
        """Copy dataset from external path to input directory."""
        if skip_copy:
            print(f"\033[33m⚠\033[0m Skipping dataset copy (--skip-copy flag)")
            return True
            
        if self.input_dataset_path.exists():
            if not force:
                print(f"\033[32m✓\033[0m Using existing dataset: {self.input_dataset_path}")
                return True  # Use existing instead of failing
            else:
                print(f"\033[33m⚠\033[0m Removing existing dataset: {self.input_dataset_path}")
                shutil.rmtree(self.input_dataset_path)
        
        print(f"\033[36m→\033[0m Copying dataset from {self.dataset_path} to {self.input_dataset_path}")
        
        try:
            shutil.copytree(self.dataset_path, self.input_dataset_path)
            print(f"\033[32m✓\033[0m Dataset copied successfully")
            return True
        except Exception as e:
            print(f"\033[31m✗\033[0m Failed to copy dataset: {e}")
            return False
    
    def prepare_output_structure(self, min_images: int = 1) -> bool:
        """Prepare or use existing output structure for the dataset."""
        # Check if output structure already exists
        if self.output_dataset_path.exists():
            print(f"\033[32m✓\033[0m Using existing output structure: {self.output_dataset_path}")
            return True
        
        print(f"\033[36m→\033[0m Preparing output structure for dataset: {self.dataset_name}")
        
        try:
            # First validate minimum images if required
            if min_images > 0:
                input_folder = INPUT_DIR / self.dataset_name
                if input_folder.exists():
                    image_count = 0
                    for ext in SUPPORTED_IMAGE_EXTENSIONS:
                        image_count += len(list(input_folder.glob(f"*{ext}")))
                        image_count += len(list(input_folder.glob(f"*{ext.upper()}")))
                    
                    if image_count < min_images:
                        print(f"\033[31m✗\033[0m Dataset has {image_count} images, minimum required: {min_images}")
                        return False
            
            # Use the single dataset preparation function
            populate_output_structure_single(self.dataset_name)
            print(f"\033[32m✓\033[0m Output structure prepared")
            return True
        except Exception as e:
            print(f"\033[31m✗\033[0m Failed to prepare output structure: {e}")
            return False
    
    def generate_config(self) -> bool:
        """Generate or use existing TOML configuration for the dataset."""
        # Check if config already exists
        if self.config_path.exists():
            print(f"\033[32m✓\033[0m Using existing configuration: {self.config_path}")
            return True
        
        print(f"\033[36m→\033[0m Generating {self.profile} configuration for dataset: {self.dataset_name}")
        
        try:
            # Generate presets for the dataset (this generates all profiles)
            generated_files = generate_presets_for_dataset(self.dataset_name)
            
            # Find the config file for our specific profile
            profile_config = None
            for config_file in generated_files:
                if config_file.parent.name == self.profile:
                    profile_config = config_file
                    break
            
            if not profile_config or not profile_config.exists():
                print(f"\033[31m✗\033[0m Configuration file not found for profile {self.profile}: {self.config_path}")
                return False
            
            print(f"\033[32m✓\033[0m Configuration generated: {profile_config}")
            # Update the config path to the actual generated file
            self.config_path = profile_config
            return True
        except Exception as e:
            print(f"\033[31m✗\033[0m Failed to generate configuration: {e}")
            return False
    
    def create_sample_prompts(self) -> bool:
        """Create or use existing sample prompts for the dataset."""
        sample_prompts_path = OUTPUT_DIR / self.dataset_name / "sample_prompts.txt"
        
        # Check if sample prompts already exist
        if sample_prompts_path.exists():
            print(f"\033[32m✓\033[0m Using existing sample prompts: {sample_prompts_path}")
            return True
        
        print(f"\033[36m→\033[0m Creating sample prompts for dataset: {self.dataset_name}")
        
        try:
            success = create_sample_prompts_for_dataset(self.dataset_name)
            if success:
                print(f"\033[32m✓\033[0m Sample prompts created")
                return True
            else:
                print(f"\033[33m⚠\033[0m Could not create sample prompts")
                return False
        except Exception as e:
            print(f"\033[31m✗\033[0m Failed to create sample prompts: {e}")
            return False
    
    def start_training(self, gpu_ids: Optional[str] = None, immediate: bool = False) -> Optional[str]:
        """Start the training job."""
        print(f"\033[36m→\033[0m Starting training for dataset: {self.dataset_name}")
        
        try:
            if immediate:
                # Run training immediately (blocking)
                print(f"\033[33m⚠\033[0m Running training immediately (blocking mode)")
                rc = run_training(self.config_path, self.profile, stream=True, gpu_ids=gpu_ids)
                if rc == 0:
                    print(f"\033[32m✓\033[0m Training completed successfully")
                    return "immediate_success"
                else:
                    print(f"\033[31m✗\033[0m Training failed with exit code: {rc}")
                    return None
            else:
                # Enqueue training job
                run_dir = compute_run_dir(self.dataset_name, self.profile)
                job = Job(self.dataset_name, self.profile, self.config_path, run_dir, gpu_ids=gpu_ids)
                JOB_MANAGER.enqueue(job)
                
                print(f"\033[32m✓\033[0m Training job enqueued: \033[33m{job.id}\033[0m")
                print(f"\033[1mOutput directory:\033[0m {run_dir}")
                return job.id
        except Exception as e:
            print(f"\033[31m✗\033[0m Failed to start training: {e}")
            return None


class SimplePipelineMonitor:
    """Simple monitoring with basic progress bar."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.should_stop = False
    
    def start_monitoring(self, refresh_interval: int = 3):
        """Start simple monitoring with progress bar."""
        print(f"\033[36m📊 Monitoring job \033[33m{self.job_id}\033[36m (Press Ctrl+C to stop)\033[0m")
        print()
        
        try:
            while not self.should_stop:
                job = self._get_current_job()
                if not job:
                    print("\nJob not found, stopping monitor")
                    break
                
                # Show status info on same line
                self._show_status_info(job)
                
                # Check if job is done
                if job.status in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED]:
                    print(f"\nJob {job.status.upper()}!")
                    break
                
                time.sleep(refresh_interval)
                    
        except KeyboardInterrupt:
            print("\nMonitoring stopped - training continues")
            print(f"Use 'autotrain train monitor --job {self.job_id}' to resume monitoring")
    
    def _get_current_job(self) -> Optional[Job]:
        """Get current job by ID."""
        for job in JOB_MANAGER.list_jobs():
            if job.id == self.job_id:
                return job
        return None
    
    def _show_status_info(self, job: Job):
        """Show current status information."""
        # Count jobs in queue
        pending_jobs = sum(1 for j in JOB_MANAGER.list_jobs() if j.status == JobStatus.PENDING)
        
        # Build status line
        status_line = ""
        
        # Simple progress bar
        if job.total_steps > 0:
            progress_percent = (job.current_step / job.total_steps) * 100
            bar_width = 30
            filled_width = int(bar_width * progress_percent / 100)
            progress_bar = "=" * filled_width + "-" * (bar_width - filled_width)
            status_line = f"[{progress_bar}] {progress_percent:.1f}% ({job.current_step}/{job.total_steps})"
        else:
            status_line = "Waiting for training to start..."
        
        # Add additional info if available
        status_parts = []
        if job.eta:
            status_parts.append(f"ETA: {job.eta}")
        if job.avg_loss:
            status_parts.append(f"Loss: {job.avg_loss:.4f}")
        if pending_jobs > 0:
            status_parts.append(f"Queue: {pending_jobs}")
        
        if status_parts:
            status_line += " | " + " | ".join(status_parts)
        
        # Print on same line, clearing previous content
        print(f"\r{status_line}\033[K", end="", flush=True)


def run_pipeline(
    dataset_path: str,
    profile: str,
    dataset_name: Optional[str] = None,
    monitor: bool = False,
    min_images: int = 1,
    gpu_ids: Optional[str] = None,
    skip_copy: bool = False,
    force: bool = False,
    dry_run: bool = False,
    immediate: bool = False,
) -> bool:
    """
    Execute the complete training pipeline.
    
    Args:
        dataset_path: Path to the external dataset
        profile: Training profile (Flux, FluxLORA, Nude)
        dataset_name: Name for the dataset (optional, derived from path)
        monitor: Start monitoring after training begins
        min_images: Minimum number of images required
        gpu_ids: GPU IDs to use for training
        skip_copy: Skip copying dataset if it already exists
        force: Force overwrite existing datasets
        dry_run: Show what would be done without executing
        immediate: Run training immediately instead of queueing
        
    Returns:
        bool: True if pipeline completed successfully
    """
    
    # Convert string path to Path object
    dataset_path_obj = Path(dataset_path).resolve()
    
    # Determine dataset name
    if not dataset_name:
        dataset_name = dataset_path_obj.name
    
    print("\033[1m" + "═" * 60 + "\033[0m")
    print("\033[1m\033[36m🚀 AutoTrain Pipeline\033[0m")
    print(f"\033[1mDataset:\033[0m \033[33m{dataset_name}\033[0m")
    print(f"\033[1mProfile:\033[0m \033[35m{profile}\033[0m") 
    print(f"\033[1mSource:\033[0m \033[32m{dataset_path_obj}\033[0m")
    print("\033[1m" + "═" * 60 + "\033[0m")
    
    # Phase 1: Validation
    print(f"\n\033[1m\033[34m╭─── Phase 1: Validation ───╮\033[0m")
    validator = PipelineValidator()
    
    validations = [
        validator.validate_dataset_path(dataset_path_obj),
        validator.validate_profile(profile),
        validator.validate_dataset_name(dataset_name),
        validator.check_disk_space(),
    ]
    
    for is_valid, message in validations:
        if is_valid:
            print(f"\033[32m✓\033[0m {message}")
        else:
            print(f"\033[31m✗\033[0m {message}")
            return False
    print("\033[1m\033[34m╰─────────────────────────────╯\033[0m")
    
    if dry_run:
        print(f"\n\033[1m\033[33m📋 DRY RUN - Showing planned operations:\033[0m")
        print(f"\033[36m1.\033[0m Copy dataset: {dataset_path_obj} → input/{dataset_name}")
        print(f"\033[36m2.\033[0m Prepare output structure for {dataset_name}")
        print(f"\033[36m3.\033[0m Generate {profile} configuration")
        print(f"\033[36m4.\033[0m Create sample prompts")
        print(f"\033[36m5.\033[0m Start training job")
        if monitor:
            print(f"\033[36m6.\033[0m Monitor training progress")
        print(f"\n\033[1m\033[32m✓ Dry run completed - no actual changes made\033[0m")
        return True
    
    # Initialize executor
    executor = PipelineExecutor(dataset_path_obj, profile, dataset_name)
    
    # Phase 2: Dataset Preparation
    print(f"\n\033[1m\033[35m╭─── Phase 2: Dataset Preparation ───╮\033[0m")
    
    if not executor.copy_dataset(force=force, skip_copy=skip_copy):
        return False
    
    if not executor.prepare_output_structure(min_images=min_images):
        return False
    print("\033[1m\033[35m╰───────────────────────────────────╯\033[0m")
    
    # Phase 3: Configuration Generation
    print(f"\n\033[1m\033[33m╭─── Phase 3: Configuration Generation ───╮\033[0m")
    
    if not executor.generate_config():
        return False
    
    # Create sample prompts (optional, don't fail if it doesn't work)
    executor.create_sample_prompts()
    print("\033[1m\033[33m╰────────────────────────────────────────╯\033[0m")
    
    # Phase 4: Training
    print(f"\n\033[1m\033[32m╭─── Phase 4: Training ───╮\033[0m")
    
    job_id = executor.start_training(gpu_ids=gpu_ids, immediate=immediate)
    if not job_id:
        return False
    print("\033[1m\033[32m╰─────────────────────────╯\033[0m")
    
    # Phase 5: Simple Monitoring (if requested)
    if monitor and job_id != "immediate_success":
        print(f"\n\033[1m\033[36m╭─── Phase 5: Monitoring ───╮\033[0m")
        
        # Use simple monitoring instead of complex train_monitor
        simple_monitor = SimplePipelineMonitor(job_id)
        simple_monitor.start_monitoring(refresh_interval=3)
        print("\033[1m\033[36m╰─────────────────────────╯\033[0m")
    
    print(f"\n\033[1m\033[32m🎉 Pipeline completed successfully!\033[0m")
    if job_id and job_id != "immediate_success":
        print(f"\033[1mJob ID:\033[0m \033[33m{job_id}\033[0m")
        print(f"\033[1mMonitor:\033[0m \033[36mautotrain train monitor --job {job_id}\033[0m")
    
    return True


def prepare_pipeline(
    dataset_path: str,
    profile: str,
    dataset_name: Optional[str] = None,
    min_images: int = 1,
    skip_copy: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Prepare dataset and configuration without starting training.
    
    This is useful for batch preparation or when you want to review
    configurations before starting training.
    """
    
    # Convert string path to Path object
    dataset_path_obj = Path(dataset_path).resolve()
    
    # Determine dataset name
    if not dataset_name:
        dataset_name = dataset_path_obj.name
    
    print("\033[1m" + "═" * 60 + "\033[0m")
    print("\033[1m\033[36m🔧 AutoTrain Pipeline - Preparation Only\033[0m")
    print(f"\033[1mDataset:\033[0m \033[33m{dataset_name}\033[0m")
    print(f"\033[1mProfile:\033[0m \033[35m{profile}\033[0m")
    print(f"\033[1mSource:\033[0m \033[32m{dataset_path_obj}\033[0m")
    print("\033[1m" + "═" * 60 + "\033[0m")
    
    # Validation
    validator = PipelineValidator()
    validations = [
        validator.validate_dataset_path(dataset_path_obj),
        validator.validate_profile(profile),
        validator.validate_dataset_name(dataset_name),
        validator.check_disk_space(),
    ]
    
    for is_valid, message in validations:
        if is_valid:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            return False
    
    if dry_run:
        print(f"\n\033[1m\033[33m📋 DRY RUN - Showing planned operations:\033[0m")
        print(f"\033[36m1.\033[0m Copy dataset: {dataset_path_obj} → input/{dataset_name}")
        print(f"\033[36m2.\033[0m Prepare output structure for {dataset_name}")
        print(f"\033[36m3.\033[0m Generate {profile} configuration")
        print(f"\033[36m4.\033[0m Create sample prompts")
        print(f"\n\033[1m\033[32m✓ Dry run completed - no actual changes made\033[0m")
        return True
    
    # Initialize executor
    executor = PipelineExecutor(dataset_path_obj, profile, dataset_name)
    
    # Execute preparation steps
    if not executor.copy_dataset(force=force, skip_copy=skip_copy):
        return False
    
    if not executor.prepare_output_structure(min_images=min_images):
        return False
    
    if not executor.generate_config():
        return False
    
    # Create sample prompts (optional)
    executor.create_sample_prompts()
    
    print(f"\n\033[1m\033[32m🔧 Dataset preparation completed!\033[0m")
    print(f"\033[1mConfiguration:\033[0m \033[33m{executor.config_path}\033[0m")
    print(f"\033[1mReady to train:\033[0m \033[36mautotrain train start --profile {executor.profile} {executor.config_path}\033[0m")
    
    return True


def discover_datasets(datasets_dir: Path) -> list[Path]:
    """
    Discover dataset directories automatically.
    
    Args:
        datasets_dir: Directory containing multiple dataset folders
        
    Returns:
        List of dataset paths that contain valid image files
    """
    discovered_datasets = []
    
    if not datasets_dir.exists() or not datasets_dir.is_dir():
        print(f"\033[31m✗\033[0m Datasets directory not found: {datasets_dir}")
        return discovered_datasets
    
    print(f"\033[36m🔍 Discovering datasets in: {datasets_dir}\033[0m")
    
    for item in datasets_dir.iterdir():
        if item.is_dir():
            # Check if directory contains image files
            image_count = 0
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_count += len(list(item.glob(f"*{ext}")))
                image_count += len(list(item.glob(f"*{ext.upper()}")))
            
            if image_count > 0:
                discovered_datasets.append(item)
                print(f"\033[32m✓\033[0m Found dataset: {item.name} ({image_count} images)")
            else:
                print(f"\033[33m⚠\033[0m Skipping {item.name} (no images found)")
    
    print(f"\033[36m📊 Discovery complete: {len(discovered_datasets)} datasets found\033[0m")
    return discovered_datasets


class BatchPipelineMonitor:
    """Enhanced monitoring for batch processing with detailed job tracking."""
    
    def __init__(self, job_ids: list[str]):
        self.job_ids = job_ids
        self.should_stop = False
        self.completed_jobs = []
        self.failed_jobs = []
    
    def start_monitoring(self, refresh_interval: int = 3):
        """Start comprehensive batch monitoring."""
        print(f"\033[36m📊 Monitoring batch processing ({len(self.job_ids)} jobs) - Press Ctrl+C to stop\033[0m")
        print()
        
        try:
            while not self.should_stop:
                jobs = self._get_all_jobs()
                
                # Update completed/failed lists
                self._update_job_lists(jobs)
                
                # Show status on same line (no screen clearing)
                self._show_batch_status(jobs)
                
                # Check if all jobs are done
                if len(self.completed_jobs) + len(self.failed_jobs) >= len(self.job_ids):
                    print("\n\033[32m🎉 All batch jobs completed!\033[0m")
                    break
                
                time.sleep(refresh_interval)
                    
        except KeyboardInterrupt:
            print("\n\033[33m⏸️  Monitoring stopped - jobs continue in background\033[0m")
            print(f"Use 'autotrain train list' to check job status")
    
    def _get_all_jobs(self) -> list[Job]:
        """Get all jobs by IDs."""
        all_jobs = JOB_MANAGER.list_jobs()
        return [job for job in all_jobs if job.id in self.job_ids]
    
    def _update_job_lists(self, jobs: list[Job]):
        """Update completed and failed job lists."""
        for job in jobs:
            if job.status == JobStatus.DONE and job.id not in [j.id for j in self.completed_jobs]:
                self.completed_jobs.append(job)
            elif job.status == JobStatus.FAILED and job.id not in [j.id for j in self.failed_jobs]:
                self.failed_jobs.append(job)
    
    def _show_batch_status(self, jobs: list[Job]):
        """Display batch status using EXACT same method as simple monitor - single line only."""
        # Find running job (should be only one)
        running_job = None
        pending_jobs = []
        
        for job in jobs:
            if job.status == JobStatus.RUNNING:
                running_job = job
            elif job.status == JobStatus.PENDING:
                pending_jobs.append(job)
        
        # Build status line - EXACTLY like SimplePipelineMonitor
        status_line = ""
        
        if running_job:
            # Simple progress bar - same as SimplePipelineMonitor
            if running_job.total_steps > 0:
                progress_percent = (running_job.current_step / running_job.total_steps) * 100
                bar_width = 30
                filled_width = int(bar_width * progress_percent / 100)
                progress_bar = "=" * filled_width + "-" * (bar_width - filled_width)
                status_line = f"🟢 {running_job.dataset}: [{progress_bar}] {progress_percent:.1f}% ({running_job.current_step}/{running_job.total_steps})"
            else:
                status_line = f"🟢 {running_job.dataset}: Waiting for training to start..."
        else:
            status_line = "No job running"
        
        # Add additional info if available - same style as SimplePipelineMonitor
        status_parts = []
        if running_job and running_job.eta:
            status_parts.append(f"ETA: {running_job.eta}")
        if running_job and running_job.avg_loss:
            status_parts.append(f"Loss: {running_job.avg_loss:.4f}")
        if pending_jobs:
            next_job = pending_jobs[0].dataset if pending_jobs else "None"
            status_parts.append(f"Queue: {len(pending_jobs)} (next: {next_job})")
        
        # Add summary
        total_jobs = len(self.job_ids)
        completed_count = len(self.completed_jobs)
        failed_count = len(self.failed_jobs)
        overall_progress = ((completed_count + failed_count) / total_jobs) * 100
        status_parts.append(f"Progress: {completed_count+failed_count}/{total_jobs} ({overall_progress:.1f}%)")
        
        if status_parts:
            status_line += " | " + " | ".join(status_parts)
        
        # Print on same line, clearing previous content - EXACT same as SimplePipelineMonitor
        print(f"\r{status_line}\033[K", end="", flush=True)


def run_batch_pipeline(
    datasets_dir: Optional[str] = None,
    dataset_paths: Optional[str] = None,
    profile: str = "FluxLORA",
    monitor: bool = False,
    min_images: int = 1,
    gpu_ids: Optional[str] = None,
    skip_copy: bool = False,
    force: bool = False,
    dry_run: bool = False,
    max_concurrent: int = 1,
) -> bool:
    """
    Execute the pipeline for multiple datasets in batch.
    
    Args:
        datasets_dir: Directory containing multiple dataset folders
        dataset_paths: Comma-separated list of dataset paths
        profile: Training profile to use for all datasets
        monitor: Start monitoring after jobs begin
        min_images: Minimum number of images required per dataset
        gpu_ids: GPU IDs to use for training
        skip_copy: Skip copying datasets if they already exist
        force: Force overwrite existing datasets
        dry_run: Show what would be done without executing
        max_concurrent: Maximum number of concurrent jobs (currently only 1 supported)
        
    Returns:
        bool: True if batch processing completed successfully
    """
    
    # Validate inputs
    if not datasets_dir and not dataset_paths:
        print("\033[31m✗\033[0m Either --datasets-dir or --dataset-paths must be provided")
        return False
    
    if datasets_dir and dataset_paths:
        print("\033[31m✗\033[0m Cannot use both --datasets-dir and --dataset-paths")
        return False
    
    # Sequential processing only for now
    if max_concurrent != 1:
        print("\033[33m⚠\033[0m Only sequential processing (max_concurrent=1) is currently supported")
        max_concurrent = 1
    
    # Discover or parse datasets
    dataset_paths_list = []
    
    if datasets_dir:
        datasets_dir_path = Path(datasets_dir).resolve()
        dataset_paths_list = discover_datasets(datasets_dir_path)
    else:
        # Parse comma-separated paths
        for path_str in dataset_paths.split(','):
            path_obj = Path(path_str.strip()).resolve()
            if path_obj.exists():
                dataset_paths_list.append(path_obj)
            else:
                print(f"\033[31m✗\033[0m Dataset path not found: {path_obj}")
                return False
    
    if not dataset_paths_list:
        print("\033[31m✗\033[0m No valid datasets found")
        return False
    
    # Display batch overview
    print("\033[1m" + "═" * 80 + "\033[0m")
    print(f"\033[1m\033[36m🚀 AutoTrain Batch Pipeline\033[0m")
    print(f"\033[1mProfile:\033[0m \033[35m{profile}\033[0m")
    print(f"\033[1mDatasets:\033[0m \033[33m{len(dataset_paths_list)}\033[0m")
    print(f"\033[1mStrategy:\033[0m \033[36mSequential Processing\033[0m")
    print("\033[1m" + "═" * 80 + "\033[0m")
    
    # List datasets
    print(f"\n\033[1m📋 Datasets to process:\033[0m")
    for i, dataset_path in enumerate(dataset_paths_list, 1):
        print(f"  {i}. \033[33m{dataset_path.name}\033[0m - {dataset_path}")
    
    if dry_run:
        print(f"\n\033[1m\033[33m📋 DRY RUN - Showing planned operations:\033[0m")
        for i, dataset_path in enumerate(dataset_paths_list, 1):
            print(f"\n\033[36mDataset {i}: {dataset_path.name}\033[0m")
            print(f"  1. Copy dataset: {dataset_path} → input/{dataset_path.name}")
            print(f"  2. Prepare output structure")
            print(f"  3. Generate {profile} configuration")
            print(f"  4. Create sample prompts")
            print(f"  5. Enqueue training job")
        
        if monitor:
            print(f"\n\033[36mAfter all jobs enqueued:\033[0m")
            print(f"  • Start batch monitoring")
        
        print(f"\n\033[1m\033[32m✓ Dry run completed - no actual changes made\033[0m")
        return True
    
    # Process datasets sequentially
    job_ids = []
    successful_preparations = 0
    
    print(f"\n\033[1m\033[35m🔄 Processing datasets...\033[0m")
    
    for i, dataset_path in enumerate(dataset_paths_list, 1):
        dataset_name = dataset_path.name
        
        print(f"\n\033[1m── Dataset {i}/{len(dataset_paths_list)}: \033[33m{dataset_name}\033[0m ──\033[0m")
        
        # Initialize executor for this dataset
        executor = PipelineExecutor(dataset_path, profile, dataset_name)
        
        # Run preparation steps
        try:
            if not executor.copy_dataset(force=force, skip_copy=skip_copy):
                print(f"\033[31m✗\033[0m Failed to prepare dataset: {dataset_name}")
                continue
            
            if not executor.prepare_output_structure(min_images=min_images):
                print(f"\033[31m✗\033[0m Failed to prepare output structure: {dataset_name}")
                continue
            
            if not executor.generate_config():
                print(f"\033[31m✗\033[0m Failed to generate configuration: {dataset_name}")
                continue
            
            # Create sample prompts (optional)
            executor.create_sample_prompts()
            
            # Enqueue training job
            job_id = executor.start_training(gpu_ids=gpu_ids, immediate=False)
            if job_id:
                job_ids.append(job_id)
                successful_preparations += 1
                print(f"\033[32m✓\033[0m Dataset {dataset_name} prepared and enqueued: \033[33m{job_id}\033[0m")
            else:
                print(f"\033[31m✗\033[0m Failed to enqueue training: {dataset_name}")
                
        except Exception as e:
            print(f"\033[31m✗\033[0m Error processing dataset {dataset_name}: {e}")
            continue
    
    # Summary
    print(f"\n\033[1m\033[32m📊 Batch Preparation Complete\033[0m")
    print(f"Successfully prepared: \033[32m{successful_preparations}\033[0m/{len(dataset_paths_list)}")
    print(f"Jobs enqueued: \033[33m{len(job_ids)}\033[0m")
    
    if not job_ids:
        print(f"\033[31m✗\033[0m No jobs were successfully enqueued")
        return False
    
    # Show enqueued jobs
    print(f"\n\033[1m🎯 Enqueued Jobs:\033[0m")
    for job_id in job_ids:
        print(f"  • \033[33m{job_id}\033[0m")
    
    # Start batch monitoring if requested
    if monitor:
        print(f"\n\033[1m\033[36m╭─── Batch Monitoring ───╮\033[0m")
        
        batch_monitor = BatchPipelineMonitor(job_ids)
        batch_monitor.start_monitoring(refresh_interval=5)
        
        print("\033[1m\033[36m╰─────────────────────────╯\033[0m")
    else:
        print(f"\n\033[1mMonitor batch:\033[0m \033[36mautotrain train list\033[0m")
    
    print(f"\n\033[1m\033[32m🎉 Batch pipeline completed!\033[0m")
    
    return True 
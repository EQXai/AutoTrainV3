from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import subprocess
import time
import os
import signal
import json
import re

from .trainer import build_accelerate_command
from .paths import OUTPUT_DIR, get_project_root, LOGS_DIR

def clear_vram():
    """Clear VRAM after job cancellation to prevent memory leaks"""
    try:
        import torch
        import gc
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Clear memory pool if available
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Log memory status
        import logging
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            logging.info(f"VRAM cleared - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
        else:
            logging.info("VRAM clearing skipped - CUDA not available")
            
    except ImportError:
        # PyTorch not available
        try:
            import gc
            gc.collect()
            import logging
            logging.info("VRAM clearing - PyTorch not available, performed basic garbage collection")
        except Exception as e:
            import logging
            logging.warning(f"VRAM clearing failed: {e}")
    except Exception as e:
        import logging
        logging.warning(f"VRAM clearing failed: {e}")

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"

class Job:
    def __init__(
        self,
        dataset: str,
        profile: str,
        toml_path: Path,
        run_dir: Path,
        *,
        gpu_ids: str | None = None,
        experiment_id: str | None = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.dataset = dataset
        self.profile = profile
        self.toml_path = toml_path
        self.run_dir = run_dir
        self.experiment_id: str | None = experiment_id
        self.gpu_ids: str | None = gpu_ids
        self.status: JobStatus = JobStatus.PENDING
        self.returncode: Optional[int] = None
        # progress stats
        self.progress_str: str = ""
        self.current_step: int = 0
        self.total_steps: int = 0
        self.percent: float = 0.0
        self.eta: str = ""
        self.elapsed: str = ""
        self.rate: float = 0.0
        self.avg_loss: float | None = None
        
        # Enhanced metrics for Phase 2
        self.learning_rate: float = 0.0
        self.train_loss: float | None = None
        self.val_loss: float | None = None
        self.memory_used: float = 0.0  # MB
        self.memory_total: float = 0.0  # MB
        self.gpu_utilization: float = 0.0  # %
        self.temperature: float = 0.0  # °C
        self.grad_norm: float | None = None
        
        # Time series data for graphs
        self.loss_history: List[Dict] = []
        self.lr_history: List[Dict] = []
        self.memory_history: List[Dict] = []
        self.gpu_history: List[Dict] = []
        
        # Training timestamps
        self.start_time: float | None = None
        self.last_update: float | None = None

class JobManager:
    def __init__(self):
        # Initialize environment variables from config file at startup
        from .utils.common import initialize_integration_env_vars
        initialize_integration_env_vars()
        
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        # live streaming vars
        self._current_job_id: Optional[str] = None
        self._live_log: str = ""
        self._live_gallery: List[str] = []
        self._current_proc: Optional[subprocess.Popen[str]] = None
        # persistence file
        self._db_path = get_project_root() / ".gradio" / "jobs.json"
        self._load_jobs()

        # Clean up any stale jobs from previous sessions
        cleaned_count = self.cleanup_stale_jobs()
        if cleaned_count > 0:
            import logging
            logging.info(f"JobManager startup: cleaned up {cleaned_count} stale job(s) from previous session")

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._stop_event = threading.Event()
        self._worker_thread.start()

    # public API
    def enqueue(self, job: Job):
        with self._lock:
            self._jobs[job.id] = job
            self._save_jobs()

        # ---------------- Google Sheets notification ----------------
        # Only try if Google Sheets is actually enabled to avoid overhead
        if os.getenv("AUTO_GSHEET_ENABLE", "0") == "1":
            try:
                from .integrations import _append_to_gsheets  # lazy import to avoid heavy dep at startup
                from .gradio_app import _load_integrations_cfg
                
                cfg = _load_integrations_cfg()
                # Check that it's configured
                if cfg.get("AUTO_GSHEET_ID"):
                    _append_to_gsheets(job)
            except Exception as e:
                # Log error for debugging but don't break queueing
                import logging
                logging.warning(f"Failed to update Google Sheets on job enqueue: {e}")

        # registry entry
        try:
            from .run_registry import upsert as _rr_upsert

            import datetime as _dt
            _rr_upsert(job, {"queued_time": _dt.datetime.utcnow().isoformat()})
        except Exception:
            pass

    def list_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str):
        """Cancel a job safely with proper state management"""
        import time
        
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            # If already completed/failed/canceled, nothing to do
            if job.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}:
                return
            
            # Case 1: Job is pending - simple cancellation
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELED
                self._save_jobs()
            
            # Case 2: Job is running - need to terminate process
            elif job.status == JobStatus.RUNNING:
                # Verify this is actually the current running job
                if self._current_job_id == job_id and self._current_proc is not None:
                    # Mark as canceled first to prevent worker from continuing
                    job.status = JobStatus.CANCELED
                    self._save_jobs()
                    
                    # Try graceful termination first
                    try:
                        if self._current_proc.poll() is None:  # Process still running
                            # Send SIGTERM to process group
                            pgid = os.getpgid(self._current_proc.pid)
                            os.killpg(pgid, signal.SIGTERM)
                            
                            # Wait a bit for graceful shutdown
                            for _ in range(30):  # Wait up to 3 seconds
                                if self._current_proc.poll() is not None:
                                    break
                                time.sleep(0.1)
                            
                            # If still running, force kill
                            if self._current_proc.poll() is None:
                                try:
                                    os.killpg(pgid, signal.SIGKILL)
                                except Exception:
                                    self._current_proc.kill()
                                    
                    except Exception:
                        # Fallback to direct process termination
                        try:
                            if self._current_proc.poll() is None:
                                self._current_proc.terminate()
                                time.sleep(0.5)
                                if self._current_proc.poll() is None:
                                    self._current_proc.kill()
                        except Exception:
                            pass
                else:
                    # Job says it's running but it's not the current job
                    # This is an inconsistent state - mark as canceled
                    job.status = JobStatus.CANCELED
                    self._save_jobs()

        # Actualizar Google Sheets con el nuevo estado cancelado
        # Only try if Google Sheets is actually enabled to avoid overhead
        if os.getenv("AUTO_GSHEET_ENABLE", "0") == "1":
            try:
                from .integrations import _append_to_gsheets
                from .gradio_app import _load_integrations_cfg
                
                cfg = _load_integrations_cfg()
                # Check that it's configured 
                if cfg.get("AUTO_GSHEET_ID"):
                    _append_to_gsheets(job)
            except Exception as e:
                # Log error for debugging but don't fail the cancellation
                import logging
                logging.warning(f"Failed to update Google Sheets on job cancel: {e}")
        
        # Clear VRAM after job cancellation to prevent memory leaks
        clear_vram()

    def cleanup_stale_jobs(self) -> int:
        """Clean up jobs with inconsistent states (e.g., marked as RUNNING but no active process)"""
        cleaned_count = 0
        
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                # Check for jobs marked as RUNNING but no current process
                if job.status == JobStatus.RUNNING and (
                    self._current_job_id != job_id or 
                    self._current_proc is None
                ):
                    # This job is marked as running but isn't actually running
                    job.status = JobStatus.FAILED
                    cleaned_count += 1
                    
                    # Log the cleanup
                    import logging
                    logging.warning(f"Cleaned up stale job {job_id}: was marked RUNNING but not active")
            
            if cleaned_count > 0:
                self._save_jobs()
        
        return cleaned_count

    def clear_queue(self, *, include_running: bool = False) -> None:
        """Remove all jobs from the manager queue.

        By default it only removes jobs that are *not* currently running.
        If *include_running* is True, the current running job will be cancelled
        and removed as well.
        """
        with self._lock:
            # Optionally cancel and remove the running job first
            if include_running and self._current_job_id:
                self.cancel(self._current_job_id)

            # Collect ids to remove (anything that is not RUNNING)
            ids_to_remove = [jid for jid, job in self._jobs.items() if job.status != JobStatus.RUNNING]

            for jid in ids_to_remove:
                self._jobs.pop(jid, None)

            # Persist changes
            self._save_jobs()
        
        # Clear VRAM after clearing queue
        clear_vram()

    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the queue (only if not running)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            # Don't remove running jobs
            if job.status == JobStatus.RUNNING:
                return False
            
            # Remove job from queue
            del self._jobs[job_id]
            
            # Save changes
            self._save_jobs()
            
            return True
    
    def clear_completed_jobs(self) -> int:
        """Clear all non-running jobs from the queue."""
        with self._lock:
            jobs_to_remove = []
            for job_id, job in self._jobs.items():
                if job.status != JobStatus.RUNNING:
                    jobs_to_remove.append(job_id)
            
            removed_count = 0
            for job_id in jobs_to_remove:
                if self.remove_job(job_id):
                    removed_count += 1
            
            return removed_count

    # internal worker
    def _worker(self):
        from pathlib import Path
        import time, os, signal, subprocess
        from .trainer import build_accelerate_command
        from .paths import OUTPUT_DIR

        while not self._stop_event.is_set():
            # pick next pending job
            with self._lock:
                next_job = next((j for j in self._jobs.values() if j.status == JobStatus.PENDING), None)
            if next_job is None:
                self._stop_event.wait(1)
                continue

            # skip if canceled before start
            with self._lock:
                if next_job.status == JobStatus.CANCELED:
                    continue
                next_job.status = JobStatus.RUNNING
                self._current_job_id = next_job.id

            # update sheet to reflect RUNNING status
            # Only try if Google Sheets is actually enabled to avoid overhead
            if os.getenv("AUTO_GSHEET_ENABLE", "0") == "1":
                try:
                    from .integrations import _append_to_gsheets
                    from .gradio_app import _load_integrations_cfg
                    
                    cfg = _load_integrations_cfg()
                    # Check that it's configured
                    if cfg.get("AUTO_GSHEET_ID"):
                        _append_to_gsheets(next_job)
                except Exception as e:
                    # Log error for debugging but don't break job execution
                    import logging
                    logging.warning(f"Failed to update Google Sheets on job start: {e}")

            # ensure run directory exists
            next_job.run_dir.mkdir(parents=True, exist_ok=True)

            # build command & paths
            cmd = build_accelerate_command(
                next_job.toml_path,
                next_job.profile,
                output_dir=next_job.run_dir,
                gpu_ids=next_job.gpu_ids,
            )
            img_dir = next_job.run_dir / "sample"

            # prepare log file
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOGS_DIR / f"{next_job.dataset}_{next_job.id}.log"

            # start process in new group
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            self._current_proc = proc
            log_lines: List[str] = []
            last_img_scan = 0.0

            try:
                while True:
                    # Check if job was canceled before processing more output
                    with self._lock:
                        if next_job.status == JobStatus.CANCELED:
                            # Job was canceled, terminate process and exit
                            try:
                                if proc.poll() is None:
                                    proc.terminate()
                                    time.sleep(0.5)
                                    if proc.poll() is None:
                                        proc.kill()
                            except Exception:
                                pass
                            break
                    
                    line = proc.stdout.readline() if proc.stdout else ""
                    if line:
                        log_lines.append(line.rstrip("\n"))
                        # append to file
                        with log_file.open("a", encoding="utf-8") as lf:
                            lf.write(line)
                    now = time.time()
                    # scan images every 2s
                    if now - last_img_scan > 2:
                        last_img_scan = now
                        gallery = sorted([str(p) for p in img_dir.glob("**/*.png")])[-12:]
                    else:
                        gallery = self._live_gallery

                    # update live vars
                    with self._lock:
                        self._live_log = "\n".join(log_lines[-400:])
                        self._live_gallery = gallery

                    # --- Enhanced metrics parsing ---
                    self._parse_enhanced_metrics(line, next_job, now)
                    
                    # --- parse progress line ---
                    prog_re = None
                    if "steps:" in line:
                        # Updated pattern to handle HH:MM:SS format for ETA
                        prog_re = re.search(r"steps:\s+(\d+)%.*\|\s*(\d+)/(\d+)\s+\[(\d+:\d\d(?::\d\d)?)<(\d+:\d\d(?::\d\d)?),\s+([0-9.]+)s/it", line)
                    if prog_re is None:
                        # fallback pattern without 'steps:' prefix - also updated for HH:MM:SS
                        prog_re = re.search(r"\s(\d+)%\|.*\s(\d+)/(\d+)\s+\[(\d+:\d\d(?::\d\d)?)<(\d+:\d\d(?::\d\d)?),\s+([0-9.]+)s/it", line)
                    if prog_re:
                        perc = int(prog_re.group(1))
                        cur = int(prog_re.group(2))
                        tot = int(prog_re.group(3))
                        elapsed = prog_re.group(4)
                        eta = prog_re.group(5)
                        rate = float(prog_re.group(6))

                        loss_match = re.search(r"avr_loss=([0-9.]+)", line)
                        loss_val = float(loss_match.group(1)) if loss_match else None

                        with self._lock:
                            # Double-check job hasn't been canceled during progress update
                            if next_job.status == JobStatus.CANCELED:
                                break
                            next_job.percent = perc
                            next_job.current_step = cur
                            next_job.total_steps = tot
                            next_job.elapsed = elapsed
                            next_job.eta = eta
                            next_job.rate = rate
                            next_job.avg_loss = loss_val
                            next_job.progress_str = f"{perc}% ETA {eta}"
                            
                            # Set start time on first progress update
                            if next_job.start_time is None:
                                next_job.start_time = now
                            next_job.last_update = now

                    if not line and proc.poll() is not None:
                        break

                rc = proc.wait()
                with self._lock:
                    next_job.returncode = rc
                    # Only update status if not already canceled
                    if next_job.status == JobStatus.RUNNING:
                        # Job completed normally
                        next_job.status = JobStatus.DONE if rc == 0 else JobStatus.FAILED
                    # If status is CANCELED, keep it as CANCELED regardless of return code
                    self._save_jobs()

                # integrations: upload model & send notifications
                try:
                    from .integrations import handle_job_complete

                    handle_job_complete(next_job)
                except Exception:  # noqa: BLE001
                    pass

                # update registry end_time
                try:
                    from .run_registry import upsert as _rr_upsert
                    import datetime as _dt
                    _rr_upsert(next_job, {"end_time": _dt.datetime.utcnow().isoformat()})
                except Exception:
                    pass

                # update registry with start_time
                try:
                    from .run_registry import upsert as _rr_upsert
                    import datetime as _dt
                    _rr_upsert(next_job, {"start_time": _dt.datetime.utcnow().isoformat()})
                except Exception:
                    pass
            except Exception as exc:
                with self._lock:
                    # Only mark as failed if not already canceled
                    if next_job.status != JobStatus.CANCELED:
                        next_job.status = JobStatus.FAILED
                    self._save_jobs()

                try:
                    from .integrations import handle_job_complete

                    handle_job_complete(next_job)
                except Exception:  # noqa: BLE001
                    pass
            finally:
                with self._lock:
                    self._current_proc = None
                    self._current_job_id = None
                    self._live_log = ""
                    self._live_gallery = []
                    self._save_jobs()
                    
                # Clear VRAM after job completion/cancellation/failure
                clear_vram()

    # --------- live helpers ---------

    def get_live_output(self, job_id: str | None = None):
        with self._lock:
            if job_id and job_id != self._current_job_id:
                return "", []
            return self._live_log, self._live_gallery

    def get_status(self, job_id: str):
        with self._lock:
            j = self._jobs.get(job_id)
            return j.status if j else None

    def get_current_job(self) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(self._current_job_id) if self._current_job_id else None

    def stop(self):
        self._stop_event.set()

    def _parse_enhanced_metrics(self, line: str, job: Job, timestamp: float):
        """Parse enhanced metrics from training log lines."""
        # Skip empty lines
        if not line.strip():
            return
        
        # Learning rate patterns (multiple variations)
        lr_patterns = [
            r"lr[:\s=]+([0-9.e-]+)",
            r"learning_rate[:\s=]+([0-9.e-]+)",
            r"LR[:\s=]+([0-9.e-]+)",
            r"learning rate[:\s=]+([0-9.e-]+)",
        ]
        
        for pattern in lr_patterns:
            lr_match = re.search(pattern, line, re.IGNORECASE)
            if lr_match:
                job.learning_rate = float(lr_match.group(1))
                job.lr_history.append({
                    "timestamp": timestamp,
                    "step": job.current_step,
                    "lr": job.learning_rate
                })
                break

        # Loss patterns (more comprehensive)
        loss_patterns = [
            (r"train[_\s]*loss[:\s=]+([0-9.]+)", "train_loss"),
            (r"training[_\s]*loss[:\s=]+([0-9.]+)", "train_loss"),
            (r"val[_\s]*loss[:\s=]+([0-9.]+)", "val_loss"),
            (r"validation[_\s]*loss[:\s=]+([0-9.]+)", "val_loss"),
            (r"loss[:\s=]+([0-9.]+)", "avg_loss"),  # Generic loss
            (r"avr_loss[:\s=]+([0-9.]+)", "avg_loss"),  # Average loss
        ]
        
        for pattern, loss_type in loss_patterns:
            loss_match = re.search(pattern, line, re.IGNORECASE)
            if loss_match:
                loss_value = float(loss_match.group(1))
                if loss_type == "train_loss":
                    job.train_loss = loss_value
                elif loss_type == "val_loss":
                    job.val_loss = loss_value
                elif loss_type == "avg_loss":
                    job.avg_loss = loss_value
                break
            
        # Update loss history if we have any loss value
        current_loss = job.train_loss or job.val_loss or job.avg_loss
        if current_loss is not None and job.current_step > 0:
            # Only add if this is a new step or significant time has passed
            if (not job.loss_history or 
                job.loss_history[-1]["step"] != job.current_step or
                timestamp - job.loss_history[-1]["timestamp"] > 10):
                
                job.loss_history.append({
                    "timestamp": timestamp,
                    "step": job.current_step,
                    "train_loss": job.train_loss,
                    "val_loss": job.val_loss,
                    "avg_loss": job.avg_loss
                })

        # Memory usage patterns (more variations)
        memory_patterns = [
            r"memory[:\s]+([0-9.]+)\s*([GM]B)",
            r"mem[:\s]+([0-9.]+)\s*([GM]B)",
            r"GPU\s+memory[:\s]+([0-9.]+)\s*([GM]B)",
            r"VRAM[:\s]+([0-9.]+)\s*([GM]B)",
        ]
        
        for pattern in memory_patterns:
            memory_match = re.search(pattern, line, re.IGNORECASE)
            if memory_match:
                memory_val = float(memory_match.group(1))
                memory_unit = memory_match.group(2).upper()
                if memory_unit == "GB":
                    memory_val *= 1024
                job.memory_used = memory_val
                break
            
        # GPU utilization patterns
        gpu_patterns = [
            r"gpu[:\s]+([0-9.]+)%",
            r"GPU[:\s]+([0-9.]+)%",
            r"utilization[:\s]+([0-9.]+)%",
            r"GPU\s+util[:\s]+([0-9.]+)%",
        ]
        
        for pattern in gpu_patterns:
            gpu_match = re.search(pattern, line, re.IGNORECASE)
            if gpu_match:
                job.gpu_utilization = float(gpu_match.group(1))
                break
            
        # Temperature patterns
        temp_patterns = [
            r"temp[:\s]+([0-9.]+)°?C",
            r"temperature[:\s]+([0-9.]+)°?C",
            r"GPU\s+temp[:\s]+([0-9.]+)°?C",
        ]
        
        for pattern in temp_patterns:
            temp_match = re.search(pattern, line, re.IGNORECASE)
            if temp_match:
                job.temperature = float(temp_match.group(1))
                break
            
        # Gradient norm patterns
        grad_patterns = [
            r"grad[_\s]*norm[:\s=]+([0-9.e-]+)",
            r"gradient[_\s]*norm[:\s=]+([0-9.e-]+)",
            r"grad_norm[:\s=]+([0-9.e-]+)",
            r"gradient_norm[:\s=]+([0-9.e-]+)",
        ]
        
        for pattern in grad_patterns:
            grad_match = re.search(pattern, line, re.IGNORECASE)
            if grad_match:
                job.grad_norm = float(grad_match.group(1))
                break

        # Update system metrics history periodically
        if (job.memory_used > 0 or job.gpu_utilization > 0) and job.current_step > 0:
            if (not job.memory_history or 
                timestamp - job.memory_history[-1]["timestamp"] > 30):  # Every 30 seconds
                
                job.memory_history.append({
                    "timestamp": timestamp,
                    "step": job.current_step,
                    "memory_used": job.memory_used,
                    "memory_total": job.memory_total,
                    "gpu_util": job.gpu_utilization,
                    "temperature": job.temperature
                })

    def get_metrics_data(self, job_id: str) -> Dict[str, Any]:
        """Get enhanced metrics data for a specific job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return {}
            
            # Always provide basic metrics from job progress
            basic_metrics = {
                "progress": job.percent,
                "current_step": job.current_step,
                "total_steps": job.total_steps,
                "eta": job.eta,
                "elapsed": job.elapsed,
                "rate": job.rate,
            }
            
            # Training metrics - include avg_loss from progress parsing
            training_metrics = {
                "learning_rate": job.learning_rate,
                "train_loss": job.train_loss,
                "val_loss": job.val_loss,
                "avg_loss": job.avg_loss,  # This comes from progress parsing
                "grad_norm": job.grad_norm,
            }
            
            # System metrics
            system_metrics = {
                "memory_used": job.memory_used,
                "memory_total": job.memory_total,
                "gpu_utilization": job.gpu_utilization,
                "temperature": job.temperature,
            }
            
            # History data
            history_data = {
                "loss_history": job.loss_history[-100:],  # Last 100 points
                "lr_history": job.lr_history[-100:],
                "memory_history": job.memory_history[-50:],  # Last 50 points
                "gpu_history": job.gpu_history[-50:],
            }
            
            # Timestamps
            timestamps = {
                "start_time": job.start_time,
                "last_update": job.last_update,
            }
            
            return {
                "basic_metrics": basic_metrics,
                "training_metrics": training_metrics,
                "system_metrics": system_metrics,
                "history_data": history_data,
                "timestamps": timestamps,
            }

    # ---------------- persistence helpers ------------------

    def _load_jobs(self):
        try:
            if self._db_path.exists():
                data = json.loads(self._db_path.read_text())
                for item in data:
                    job = Job(
                        item["dataset"],
                        item["profile"],
                        Path(item["toml_path"]),
                        Path(item["run_dir"]),
                        gpu_ids=item.get("gpu_ids"),
                        experiment_id=item.get("experiment_id"),
                    )
                    job.id = item["id"]
                    job.status = JobStatus(item["status"])
                    # if it was running when app closed, mark as failed
                    if job.status == JobStatus.RUNNING:
                        job.status = JobStatus.FAILED
                    self._jobs[job.id] = job
        except Exception:
            pass

    def _save_jobs(self):
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "id": j.id,
                    "dataset": j.dataset,
                    "profile": j.profile,
                    "toml_path": str(j.toml_path),
                    "run_dir": str(j.run_dir),
                    "status": j.status,
                    "experiment_id": j.experiment_id,
                    "gpu_ids": j.gpu_ids,
                }
                for j in self._jobs.values()
            ]
            self._db_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Default singleton instance
# ---------------------------------------------------------------------------

# Creating it here avoids circular imports (experiments -> job_manager)
JOB_MANAGER = JobManager() 
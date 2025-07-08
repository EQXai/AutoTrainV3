#!/usr/bin/env python3
"""
Script para limpiar la base de datos de jobs y resetear el estado
"""

import json
from pathlib import Path
import sys
import shutil
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from autotrain_sdk.paths import get_project_root
from autotrain_sdk.job_manager import JobManager, JobStatus

def backup_jobs_db():
    """Crear backup de la base de datos de jobs"""
    jm = JobManager()
    db_path = jm._db_path
    
    if db_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.parent / f"jobs_backup_{timestamp}.json"
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    else:
        print("ℹ️  No jobs database found to backup")
        return None

def analyze_current_jobs():
    """Analizar los jobs actuales"""
    print("🔍 Analyzing current jobs...")
    
    jm = JobManager()
    jobs = jm.list_jobs()
    
    print(f"📊 Total jobs: {len(jobs)}")
    
    # Count by status
    status_counts = {}
    for job in jobs:
        status = job.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("📈 Jobs by status:")
    for status, count in status_counts.items():
        print(f"   {status}: {count}")
    
    # Show recent jobs
    print("\n📋 Last 10 jobs:")
    for i, job in enumerate(jobs[-10:], 1):
        print(f"   {i:2d}. {job.id} | {job.dataset} | {job.status.value}")
    
    # Check for problematic jobs
    print("\n🔍 Checking for problematic jobs...")
    current_job_id = jm._current_job_id
    current_proc = jm._current_proc is not None
    
    print(f"   Current job ID: {current_job_id}")
    print(f"   Current process exists: {current_proc}")
    
    # Find stale jobs
    stale_jobs = []
    for job in jobs:
        if job.status == JobStatus.RUNNING and job.id != current_job_id:
            stale_jobs.append(job)
    
    if stale_jobs:
        print(f"   ⚠️  Found {len(stale_jobs)} stale RUNNING jobs:")
        for job in stale_jobs:
            print(f"      - {job.id} | {job.dataset}")
    else:
        print("   ✅ No stale jobs found")
    
    jm.stop()
    return len(jobs), status_counts, stale_jobs

def clean_completed_jobs():
    """Limpiar jobs completados/fallidos/cancelados antiguos"""
    print("\n🧹 Cleaning completed jobs...")
    
    jm = JobManager()
    jobs = jm.list_jobs()
    original_count = len(jobs)
    
    # Keep only recent completed jobs and all pending/running jobs
    jobs_to_keep = []
    jobs_to_remove = []
    
    # Separate by status
    pending_running = [j for j in jobs if j.status in {JobStatus.PENDING, JobStatus.RUNNING}]
    completed = [j for j in jobs if j.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELED}]
    
    print(f"   Pending/Running jobs: {len(pending_running)} (will keep all)")
    print(f"   Completed jobs: {len(completed)}")
    
    # Keep all pending/running
    jobs_to_keep.extend(pending_running)
    
    # Keep only last 10 completed jobs
    completed_sorted = sorted(completed, key=lambda x: x.id)  # Sort by ID (chronological)
    jobs_to_keep.extend(completed_sorted[-10:])  # Keep last 10
    jobs_to_remove.extend(completed_sorted[:-10])  # Remove older ones
    
    print(f"   Keeping {len(jobs_to_keep)} jobs")
    print(f"   Removing {len(jobs_to_remove)} old completed jobs")
    
    # Update the jobs dictionary
    with jm._lock:
        jm._jobs = {job.id: job for job in jobs_to_keep}
        jm._save_jobs()
    
    print(f"✅ Cleaned database: {original_count} → {len(jobs_to_keep)} jobs")
    
    jm.stop()
    return len(jobs_to_remove)

def force_cleanup_stale_jobs():
    """Forzar limpieza de jobs con estados inconsistentes"""
    print("\n🔧 Force cleaning stale jobs...")
    
    jm = JobManager()
    cleaned = jm.cleanup_stale_jobs()
    print(f"✅ Cleaned {cleaned} stale jobs")
    
    jm.stop()
    return cleaned

def reset_current_job_state():
    """Resetear el estado del job actual si hay inconsistencias"""
    print("\n🔄 Resetting current job state...")
    
    jm = JobManager()
    
    with jm._lock:
        # Clear any current job reference if no process is running
        if jm._current_job_id and jm._current_proc is None:
            print(f"   Clearing stale current job ID: {jm._current_job_id}")
            jm._current_job_id = None
        
        # Clear live data
        jm._live_log = ""
        jm._live_gallery = []
        
        # Save state
        jm._save_jobs()
    
    print("✅ Current job state reset")
    jm.stop()

def main():
    """Proceso principal de limpieza"""
    print("🚀 AutoTrainV2 Jobs Database Cleanup")
    print("=" * 50)
    
    # Step 1: Backup
    backup_path = backup_jobs_db()
    
    # Step 2: Analyze current state
    total_jobs, status_counts, stale_jobs = analyze_current_jobs()
    
    # Step 3: Ask user what to do
    print(f"\n{'='*50}")
    print("🛠️  Cleanup Options:")
    print("1. Light cleanup - Remove old completed jobs only")
    print("2. Force cleanup - Fix stale states + remove old jobs")  
    print("3. Nuclear option - Clear ALL jobs (except current running)")
    print("4. Just fix stale states")
    print("5. Cancel - do nothing")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            # Light cleanup
            removed = clean_completed_jobs()
            print(f"\n✅ Light cleanup completed. Removed {removed} old jobs.")
            
        elif choice == "2":
            # Force cleanup
            stale_cleaned = force_cleanup_stale_jobs()
            removed = clean_completed_jobs()
            reset_current_job_state()
            print(f"\n✅ Force cleanup completed. Fixed {stale_cleaned} stale jobs, removed {removed} old jobs.")
            
        elif choice == "3":
            # Nuclear option
            print("\n⚠️  NUCLEAR OPTION: This will remove ALL jobs except currently running ones!")
            confirm = input("Type 'DELETE ALL' to confirm: ").strip()
            
            if confirm == "DELETE ALL":
                jm = JobManager()
                with jm._lock:
                    # Keep only currently running job
                    if jm._current_job_id and jm._current_job_id in jm._jobs:
                        current_job = jm._jobs[jm._current_job_id]
                        jm._jobs = {jm._current_job_id: current_job}
                        print(f"   Kept current running job: {jm._current_job_id}")
                    else:
                        jm._jobs = {}
                        jm._current_job_id = None
                        print("   Cleared all jobs - no current running job found")
                    
                    jm._save_jobs()
                jm.stop()
                print("💥 Nuclear cleanup completed!")
            else:
                print("❌ Nuclear cleanup cancelled")
                
        elif choice == "4":
            # Just fix stale states
            stale_cleaned = force_cleanup_stale_jobs()
            reset_current_job_state()
            print(f"\n✅ Fixed {stale_cleaned} stale job states")
            
        elif choice == "5":
            print("❌ Cleanup cancelled")
            
        else:
            print("❌ Invalid option")
            
    except KeyboardInterrupt:
        print("\n❌ Cleanup cancelled by user")
    
    # Final analysis
    print(f"\n{'='*50}")
    print("📊 Final State:")
    final_total, final_status, final_stale = analyze_current_jobs()
    
    print(f"\n🏁 Cleanup completed!")
    if backup_path:
        print(f"💾 Backup available at: {backup_path}")

if __name__ == "__main__":
    main() 
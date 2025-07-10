# AutoTrain Pipeline Documentation

## Overview

The `pipeline.py` module provides a unified command-line interface for automating the complete AI model training process, from external dataset preparation to live training monitoring.

## Pipeline Architecture

The pipeline is divided into **5 main phases**:

1. **🔍 Validation**: Verifies that datasets, profiles, and system requirements are valid
2. **📁 Dataset Preparation**: Copies and structures input data
3. **⚙️ Configuration Generation**: Creates TOML configuration files
4. **🚀 Training**: Starts training jobs
5. **📊 Monitoring**: Supervises training progress

## Operation Modes

### 1. `run` Mode - Individual Processing

Processes a single dataset step by step.

```bash
autotrain pipeline run --dataset-path /path/to/dataset --profile Flux --monitor
```

**Features:**
- Processes a single dataset
- Sequential execution of all phases
- Optional real-time monitoring
- Ideal for individual datasets or testing

### 2. `batch` Mode - Batch Processing

Processes multiple datasets automatically.

```bash
# Process all datasets in a directory
autotrain pipeline batch --datasets-dir /path/to/datasets --profile FluxLORA --monitor

# Process specific datasets
autotrain pipeline batch --dataset-paths "/path/to/dataset1,/path/to/dataset2" --profile Nude
```

**Features:**
- Processes multiple datasets automatically
- Automatic dataset discovery
- Sequential job queuing
- Batch monitoring with global progress
- Ideal for mass processing

### 3. `prepare` Mode - Preparation Only

Prepares datasets without starting training.

```bash
autotrain pipeline prepare --dataset-path /path/to/dataset --profile Flux
```

**Features:**
- Only executes preparation phases (1-3)
- Does not start training
- Useful for prior preparation or configuration review

## Command Line Arguments

### Main Arguments

| Argument | Description | Required | Example |
|----------|-------------|----------|---------|
| `--dataset-path` | Path to individual dataset (run mode) | ✅ (run) | `--dataset-path /data/my_dataset` |
| `--datasets-dir` | Directory with multiple datasets (batch mode) | ✅ (batch) | `--datasets-dir /data/datasets/` |
| `--dataset-paths` | Comma-separated list of paths (batch mode) | ✅ (batch) | `--dataset-paths "/data/set1,/data/set2"` |
| `--profile` | Training profile | ✅ | `--profile Flux` |

### Available Profiles

- **`Flux`**: Complete training with Flux model
- **`FluxLORA`**: LoRA training with Flux model
- **`Nude`**: Specialized profile for NSFW content

### Control Arguments

| Argument | Description | Type | Default | Example |
|----------|-------------|------|---------|---------|
| `--dataset-name` | Custom name for the dataset | `str` | `auto` | `--dataset-name my_model` |
| `--min-images` | Minimum required images | `int` | `1` | `--min-images 10` |
| `--gpu-ids` | Specific GPUs to use | `str` | `auto` | `--gpu-ids "0,1"` |
| `--max-concurrent` | Maximum concurrent jobs (batch only) | `int` | `1` | `--max-concurrent 2` |

### Behavior Flags

| Flag | Description | Effect |
|------|-------------|--------|
| `--monitor` | Enable real-time monitoring | Shows training progress |
| `--skip-copy` | Skip copy if dataset already exists | Speeds up process |
| `--force` | Force overwrite existing files | Replaces existing datasets |
| `--dry-run` | Show operations without executing | Test mode |
| `--immediate` | Execute training immediately | Synchronous mode (blocks terminal) |

## Usage Examples

### Example 1: Basic Individual Training

```bash
# Train an individual dataset with monitoring
autotrain pipeline run \
  --dataset-path /home/user/datasets/character_photos \
  --profile FluxLORA \
  --monitor
```

### Example 2: Batch Processing

```bash
# Process all datasets in a directory
autotrain pipeline batch \
  --datasets-dir /home/user/all_datasets \
  --profile Flux \
  --min-images 5 \
  --monitor
```

### Example 3: Prior Preparation

```bash
# Only prepare configurations for review
autotrain pipeline prepare \
  --dataset-path /home/user/datasets/test_character \
  --profile Nude \
  --dry-run
```

### Example 4: Advanced Configuration

```bash
# Training with specific configuration
autotrain pipeline run \
  --dataset-path /data/premium_dataset \
  --profile Flux \
  --dataset-name premium_model_v2 \
  --gpu-ids "0,1" \
  --min-images 20 \
  --force \
  --immediate
```

## Detailed Workflow

### Phase 1: Validation 🔍

```
✓ Verify dataset exists and contains images
✓ Validate profile compatibility
✓ Check available disk space
✓ Validate dataset name format
```

### Phase 2: Dataset Preparation 📁

```
→ Copy dataset to input/ directory
→ Create output/ directory structure
→ Validate minimum image count
```

### Phase 3: Configuration Generation ⚙️

```
→ Generate TOML configuration file
→ Create sample prompts
→ Configure training parameters
```

### Phase 4: Training 🚀

```
→ Queue training job
→ Assign GPU resources
→ Start training process
```

### Phase 5: Monitoring 📊

```
→ Show real-time progress
→ Report training metrics
→ Estimate remaining time
```

## File Structure

```
AutoTrainV2/
├── input/                    # Input datasets
│   └── dataset_name/
│       ├── image_01.jpg
│       ├── image_01.txt
│       └── ...
├── output/                   # Training results
│   └── dataset_name/
│       ├── model/           # Trained model
│       ├── log/             # Training logs
│       └── sample_prompts.txt
├── BatchConfig/              # Profile configurations
│   ├── Flux/
│   ├── FluxLORA/
│   └── Nude/
└── logs/                     # System logs
```

## Monitoring and Tracking

### Individual Monitoring

```bash
# During training shows:
[============================--] 93.5% (1870/2000) | ETA: 00:05:23 | Loss: 0.0234 | Queue: 0
```

### Batch Monitoring

```bash
# For batch processing:
🟢 dataset_character: [==============----] 70.0% (350/500) | ETA: 00:10:15 | Loss: 0.0156 | Queue: 2 (next: dataset_style) | Progress: 2/5 (40.0%)
```

### Tracking Commands

```bash
# List all jobs
autotrain train list

# Monitor specific job
autotrain train monitor --job <job_id>

# Cancel job
autotrain train cancel --job <job_id>
```

## Troubleshooting

### Common Errors

1. **"No image files found"**
   - Verify dataset contains `.jpg`, `.png`, etc. files
   - Use `--min-images 0` for test datasets

2. **"Invalid profile"**
   - Verify profile is one of: `Flux`, `FluxLORA`, `Nude`

3. **"Insufficient disk space"**
   - Free up disk space (~5GB required by default)

4. **"Configuration file not found"**
   - Verify templates exist in `templates/`

### Logs and Debugging

```bash
# View error logs
tail -f logs/autotrain_errors.log

# View pipeline logs
tail -f logs/autotrain_pipeline.log
```

## Advanced Customization

### Creating Custom Profiles

1. Create template in `templates/MyProfile/base.toml`
2. Add `MyProfile` to `VALID_PROFILES` in `pipeline.py`
3. Use with `--profile MyProfile`

### Script Integration

```python
from autotrain_sdk.pipeline import run_pipeline, run_batch_pipeline

# Use programmatically
success = run_pipeline(
    dataset_path="/path/to/dataset",
    profile="FluxLORA",
    monitor=True,
    min_images=10
)
```

## Best Practices

1. **Use `--dry-run`** before processing large datasets
2. **Validate datasets** with appropriate `--min-images`
3. **Use `--skip-copy`** to save time on re-training
4. **Monitor resources** with specific `--gpu-ids`
5. **Backup important configurations**
6. **Review logs** regularly to detect issues

---

*This pipeline is designed to be robust and efficient, automating the complete AI model training process with minimal manual intervention.* 
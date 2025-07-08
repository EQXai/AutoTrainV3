# AutoTrainV2

AutoTrainV2 is a Python/Bash toolkit that simplifies dataset management and training of Stable-Diffusion models (Flux, FluxLORA, Nude) via the `sd-scripts` project.

It exposes two fully-featured front-ends:

* **Gradio Web UI** – graphical, runs in the browser.
* **Command-line interface (CLI)** – scriptable, perfect for automation and low-resource machines.

---

## 1. Installation

```bash
# 1. Clone the repo
$ git clone https://github.com/yourname/AutoTrainV2.git
$ cd AutoTrainV2

# 2. (Recommended) Create a virtual environment
$ python3 -m venv venv
$ source venv/bin/activate

# 3. Install dependencies
$ pip install -r requirements.txt
```

> **GPU support** – make sure you have CUDA, cuDNN and the appropriate PyTorch wheel installed if you want to train on NVIDIA GPUs.  The project also works on CPU-only machines (slow!).

---

## 2. Directory layout (after first run)

```
AutoTrainV2/
├─ input/             # raw datasets – one sub-folder per dataset
├─ output/            # generated samples, logs and model checkpoints
├─ BatchConfig/       # automatically generated *.toml* presets
├─ logs/              # CLI/queue logs (*.log)
└─ templates/         # base templates for each training profile
```

---

## 3. Using the Gradio Web UI

1. **Launch the server**

   ```bash
   $ python -m autotrain_sdk.gradio_app  # or
   $ autotrain web serve --share         # adds a public "share" link
   ```

2. **Navigate** to the printed URL (default http://127.0.0.1:7860) and explore the tabs:

   * **Dataset** – create dataset folders, import external datasets and build the *output/* structure.
   * **Config** – view / edit *.toml* presets in an interactive table.
   * **Training** – enqueue training jobs for Flux / FluxLORA / Nude profiles.
   * **Queue** – real-time job list, cancel button and the newly-added **Clear queue** button.
   * **Model Organizer** – browse finished runs, download as ZIP, rename, delete or upload to HF.
   * **Integrations** – tokens and settings for HF Hub, S3, webhooks, SMTP, remote output, …

3. **Live progress** – once a job starts you will see streaming logs and sample images in the "Training" tab; the "Queue" tab keeps aggregated statistics.

---

## 4. Using the CLI

The CLI is declared via *Typer*; every command offers `--help` for details.

### 4.1 Dataset commands

```bash
# Create two dataset folders inside input/
autotrain dataset create --names "alex,maria"

# Build the output/ structure, copying images, and require ≥20 JPG/PNG per dataset
autotrain dataset build-output --min-images 20

# Clean workspace (input/, output/, BatchConfig/)
autotrain dataset clean --no-output   # fine-grained flags available
```

### 4.2 Config commands

```bash
# Regenerate all presets (.toml) from templates
autotrain config refresh

# Show a preset as a table
autotrain config show BatchConfig/Flux/alex.toml

# Override multiple keys
autotrain config set BatchConfig/Flux/alex.toml \
    --kv lr=1e-5 --kv train_batch_size=2
```

### 4.3 Training commands

```bash
# Enqueue a job (non-blocking)
autotrain train start --profile Flux --file BatchConfig/Flux/alex.toml

# Run immediately and stream logs
autotrain train start --profile FluxLORA --file BatchConfig/Flux/alex.toml --now

# Restrict to specific GPUs
autotrain train start --gpu 0,1 --profile Nude --file BatchConfig/Nude/alex.toml
```

### 4.4 Web UI command

```bash
# Same as python -m autotrain_sdk.gradio_app
autotrain web serve --share  # optional public tunnel
```

---

## 5. Job queue lifecycle (CLI shortcuts)

The CLI currently manipulates the queue implicitly via `train start`.  For advanced queue management (refresh, cancel, clear) use the **Queue** tab in Gradio.

---

## 6. Environment variables

| Variable               | Purpose                                              |
|------------------------|------------------------------------------------------|
| `CUDA_VISIBLE_DEVICES` | Limit GPUs used by training & queue worker           |
| `AUTO_HF_TOKEN`        | Hugging Face access token (save in Integrations tab) |
| `AUTO_REMOTE_BASE`     | Remote filesystem path for outputs (eg. rclone)      |
| `AUTO_REMOTE_DIRECT`   | `1` → write outputs directly to remote path          |

---

## 7. Development & Contributing

1. Commit hooks: `pre-commit install`
2. Run unit tests: `pytest -q`
3. Lint: `ruff check .`

Feel free to open issues or pull requests! 
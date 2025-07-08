# AutoTrainV2

AutoTrainV2 is a Python toolkit that simplifies training of Stable Diffusion models (Flux, FluxLORA, SDXL) using the `sd-scripts` framework. It provides three user interfaces: Gradio Web UI, CLI, and Interactive Menu.

---

## 🛠️ Installation

### Requirements
- **OS**: Linux (Ubuntu 20.04+) or Windows with WSL2
- **GPU**: NVIDIA with 8GB+ VRAM
- **Python**: 3.9+
- **CUDA**: 12.8+ with compatible drivers

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourname/AutoTrainV2.git
cd AutoTrainV2

# 2. Run the automated setup script
bash setup.sh
```

The setup script will automatically:
- ✅ Create virtual environment
- ✅ Install PyTorch with CUDA 12.8
- ✅ Install xformers
- ✅ Install all dependencies
- ✅ Download base Flux models
- ✅ Verify installation

---

## 📁 Project Structure

After installation, your project structure will be:

```
AutoTrainV2/
├── 📁 input/                    # Raw datasets (place your images here)
│   └── dataset_name/
│       ├── image_01.jpg         # Training images
│       ├── image_01.txt         # Caption for image_01.jpg
│       └── ...
│
├── 📁 output/                   # Processed training structure
│   └── dataset_name/
│       ├── img/                 # Processed images for training
│       ├── model/               # Output trained models
│       ├── log/                 # Training logs
│       └── sample_prompts.txt   # Generation prompts
│
├── 📁 BatchConfig/              # Training configurations
│   ├── Flux/                    # Flux checkpoint training configs
│   ├── FluxLORA/               # Flux LoRA training configs
│   └── Nude/                   # SDXL Nude training configs
│
├── 📁 models/                   # Base models
│   └── trainX/                  # Flux base models
│
├── 📁 templates/               # Configuration templates
└── 📁 autotrain_sdk/          # Main source code
```

---

## 🚀 Usage

AutoTrainV2 provides three ways to use the system:

### 1. Web Interface (Recommended)

Start the Gradio web interface:

```bash
# Activate environment
source venv/bin/activate

# Start web server with public sharing
python -m autotrain_sdk web serve --share
```

Open your browser to: http://127.0.0.1:7860

**Web Interface Tabs:**
- **Dataset**: Create and manage training datasets
- **Config**: Edit training configurations visually
- **Training**: Start and monitor training jobs
- **Queue**: View active training queue
- **Model Organizer**: Manage completed models
- **Integrations**: Configure external services

### 2. Interactive Menu

For guided, step-by-step usage:

```bash
# Activate environment
source venv/bin/activate

# Start Gradio UI
python -m autotrain_sdk web serve --share

# Start CLI menu
python -m autotrain_sdk.menu
```

---
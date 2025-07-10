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

### 2. Automated Pipeline (NEW!)

For complete automation from dataset to training:

```bash
# Activate environment
source venv/bin/activate

# Complete pipeline in one command
python -m autotrain_sdk pipeline run --dataset-path /path/to/dataset --profile Flux --monitor

# Equivalent to all manual steps:
# 1. Copy dataset to input/
# 2. Build output structure
# 3. Generate configuration
# 4. Start training
# 5. Monitor progress
```

### 3. Interactive Menu

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

## 🚀 Pipeline Automatizado (Nuevo!)

El pipeline automatizado permite ejecutar todo el proceso de entrenamiento con un solo comando:

### Comando Básico
```bash
# Entrenamiento completo automatizado
python -m autotrain_sdk pipeline run --dataset-path /path/to/dataset --profile Flux --monitor
```

### Opciones Avanzadas
```bash
# Con GPU específica y nombre personalizado
python -m autotrain_sdk pipeline run \
  --dataset-path /path/to/dataset \
  --profile FluxLORA \
  --dataset-name my_model \
  --gpu 0,1 \
  --monitor

# Solo preparación (sin entrenar)
python -m autotrain_sdk pipeline prepare --dataset-path /path/to/dataset --profile Flux

# Dry run (mostrar plan sin ejecutar)
python -m autotrain_sdk pipeline run --dataset-path /path/to/dataset --profile Flux --dry-run

# Saltar copia si dataset ya existe
python -m autotrain_sdk pipeline run --dataset-path /path/to/dataset --profile Flux --skip-copy
```

### Ventajas del Pipeline
- **Un solo comando** reemplaza múltiples pasos manuales
- **Validación automática** previene errores comunes
- **Monitoreo integrado** sin intervención manual
- **Manejo de errores** robusto con mensajes claros

---
#!/bin/bash

# 🚀 AutoTrainV2 - Script de Flujo de Trabajo Rápido
# Este script replica el flujo de trabajo solicitado por el cliente
# pero usando los comandos CLI avanzados de AutoTrainV2

set -e  # Salir si cualquier comando falla

# ====================================================================
# CONFIGURACIÓN
# ====================================================================

# Configuración del cliente (modificar según necesidades)
SOURCE_DIR="/workspace/datasets"
TARGET_DIR="input"
DATASETS=("ExampleFolder1" "ExampleFolder2" "ExampleFolder3")
PROFILE="Flux"  # Opciones: Flux, FluxLORA, Nude
GPU_ID="0"      # GPU a utilizar
PRIORITY="high" # Prioridad: low, normal, high

# Configuración avanzada
MIN_IMAGES=1
REFRESH_INTERVAL=3
AUTO_MONITOR=true

# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

print_step() {
    echo "=================================================="
    echo "🔄 $1"
    echo "=================================================="
}

print_info() {
    echo "ℹ️  $1"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
    exit 1
}

# Función para ejecutar comandos autotrain
run_autotrain() {
    if command -v autotrain &> /dev/null; then
        autotrain "$@"
    else
        python -m autotrain_sdk "$@"
    fi
}

check_requirements() {
    print_step "Verificando Requisitos"
    
    # Verificar que estamos en el directorio correcto
    if [ ! -f "pyproject.toml" ]; then
        print_error "No se encontró pyproject.toml. Asegúrate de estar en el directorio AutoTrainV2"
    fi
    
    # Verificar entorno virtual
    if [ ! -d "venv" ]; then
        print_error "No se encontró el entorno virtual. Ejecuta 'bash setup.sh' primero"
    fi
    
    # Activar entorno virtual
    source venv/bin/activate
    
    # Verificar comando autotrain o módulo Python
    if ! command -v autotrain &> /dev/null; then
        print_info "Comando 'autotrain' no instalado, usando python -m autotrain_sdk"
        # Verificar que el módulo Python funciona
        if ! python -c "import autotrain_sdk.cli" &> /dev/null; then
            print_error "Módulo autotrain_sdk no encontrado. Verifica la instalación"
        fi
    fi
    
    print_success "Requisitos verificados"
}

import_datasets() {
    print_step "Importando Datasets"
    
    # Crear estructura de carpetas primero
    dataset_names=$(IFS=,; echo "${DATASETS[*]}")
    print_info "Creando estructura para: $dataset_names"
    run_autotrain dataset create --names "$dataset_names"
    
    # Importar cada dataset
    for dataset in "${DATASETS[@]}"; do
        print_info "Procesando dataset: $dataset"
        
        if [ -d "$SOURCE_DIR/$dataset" ]; then
            # Método 1: Copia directa (como solicita el cliente)
            print_info "  Copiando desde $SOURCE_DIR/$dataset"
            cp -r "$SOURCE_DIR/$dataset" "$TARGET_DIR/"
            
            # Verificar copia
            if [ -d "$TARGET_DIR/$dataset" ]; then
                img_count=$(find "$TARGET_DIR/$dataset" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
                print_success "  $dataset copiado exitosamente ($img_count imágenes)"
            else
                print_error "  Error copiando $dataset"
            fi
        else
            print_error "  Dataset $dataset no encontrado en $SOURCE_DIR"
        fi
    done
    
    print_success "Importación de datasets completada"
}

build_structure() {
    print_step "Construyendo Estructura de Entrenamiento"
    
    print_info "Creando estructura en directorio 'output/'"
    run_autotrain dataset build-output --min-images $MIN_IMAGES
    
    # Verificar estructura creada
    print_info "Verificando estructura creada:"
    for dataset in "${DATASETS[@]}"; do
        if [ -d "output/$dataset" ]; then
            img_count=$(find "output/$dataset/img" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
            print_success "  $dataset: $img_count imágenes procesadas"
        else
            print_error "  Error: estructura no creada para $dataset"
        fi
    done
    
    print_success "Estructura de entrenamiento creada"
}

create_prompts() {
    print_step "Creando Sample Prompts"
    
    print_info "Generando archivos sample_prompts.txt"
    run_autotrain dataset create-prompts
    
    # Verificar prompts creados
    print_info "Verificando prompts creados:"
    for dataset in "${DATASETS[@]}"; do
        if [ -f "output/$dataset/sample_prompts.txt" ]; then
            lines=$(wc -l < "output/$dataset/sample_prompts.txt")
            print_success "  $dataset: $lines prompts generados"
        else
            print_error "  Error: prompts no creados para $dataset"
        fi
    done
    
    print_success "Sample prompts creados"
}

generate_presets() {
    print_step "Generando Presets de Configuración"
    
    print_info "Regenerando presets TOML (equivalente a python generate_preset5.py)"
    run_autotrain config refresh
    
    # Verificar presets generados
    print_info "Verificando presets generados:"
    for dataset in "${DATASETS[@]}"; do
        preset_file="BatchConfig/$PROFILE/$dataset.toml"
        if [ -f "$preset_file" ]; then
            print_success "  $dataset: preset generado en $preset_file"
        else
            print_error "  Error: preset no generado para $dataset"
        fi
    done
    
    print_success "Presets de configuración generados"
}

start_training() {
    print_step "Iniciando Entrenamiento"
    
    print_info "Iniciando entrenamiento por lotes (equivalente a python run_training.py)"
    dataset_names=$(IFS=,; echo "${DATASETS[*]}")
    
    # Iniciar entrenamiento por lotes
    run_autotrain train batch \
        --datasets "$dataset_names" \
        --profile "$PROFILE" \
        --gpu "$GPU_ID" \
        --priority "$PRIORITY"
    
    print_success "Entrenamiento iniciado para: $dataset_names"
    
    # Mostrar estado de la cola
    print_info "Estado actual de la cola:"
    run_autotrain train queue list
}

monitor_training() {
    if [ "$AUTO_MONITOR" = true ]; then
        print_step "Iniciando Monitoreo"
        
        print_info "Iniciando monitoreo en tiempo real"
        print_info "Presiona Ctrl+C para salir del monitoreo"
        
        # Dar tiempo para que se inicialice el entrenamiento
        sleep 3
        
        # Iniciar monitoreo
        run_autotrain train monitor --refresh "$REFRESH_INTERVAL"
    else
        print_info "Monitoreo automático deshabilitado"
        print_info "Para monitorear manualmente usa: python -m autotrain_sdk train monitor"
    fi
}

show_summary() {
    print_step "Resumen del Proceso"
    
    print_info "Datasets procesados: $(IFS=,; echo "${DATASETS[*]}")"
    print_info "Perfil utilizado: $PROFILE"
    print_info "GPU asignada: $GPU_ID"
    print_info "Prioridad: $PRIORITY"
    
    print_info "Comandos útiles para monitoreo:"
    echo "  • Ver cola: python -m autotrain_sdk train queue list"
    echo "  • Monitorear: python -m autotrain_sdk train monitor"
    echo "  • Ver logs: python -m autotrain_sdk train logs --follow"
    echo "  • Ver estado: python -m autotrain_sdk train status --verbose"
    echo "  • Historial: python -m autotrain_sdk train history"
    
    print_success "Proceso completado exitosamente"
}

# ====================================================================
# MENÚ INTERACTIVO OPCIONAL
# ====================================================================

interactive_setup() {
    echo "🎯 Configuración Interactiva"
    echo "Presiona Enter para usar valores por defecto o ingresa nuevos valores:"
    echo
    
    read -p "Directorio fuente [$SOURCE_DIR]: " input_source
    SOURCE_DIR=${input_source:-$SOURCE_DIR}
    
    read -p "Perfil de entrenamiento [$PROFILE] (Flux/FluxLORA/Nude): " input_profile
    PROFILE=${input_profile:-$PROFILE}
    
    read -p "GPU ID [$GPU_ID]: " input_gpu
    GPU_ID=${input_gpu:-$GPU_ID}
    
    read -p "Prioridad [$PRIORITY] (low/normal/high): " input_priority
    PRIORITY=${input_priority:-$PRIORITY}
    
    read -p "¿Iniciar monitoreo automático? [$AUTO_MONITOR] (true/false): " input_monitor
    AUTO_MONITOR=${input_monitor:-$AUTO_MONITOR}
    
    echo
    echo "Configuración actualizada:"
    echo "  Directorio fuente: $SOURCE_DIR"
    echo "  Perfil: $PROFILE"
    echo "  GPU: $GPU_ID"
    echo "  Prioridad: $PRIORITY"
    echo "  Monitoreo automático: $AUTO_MONITOR"
    echo
    
    read -p "¿Continuar con esta configuración? (y/n): " confirm
    if [[ $confirm != [yY] ]]; then
        echo "Operación cancelada"
        exit 0
    fi
}

# ====================================================================
# FUNCIÓN PRINCIPAL
# ====================================================================

main() {
    echo "🚀 AutoTrainV2 - Flujo de Trabajo Automatizado"
    echo "Este script replica el flujo de trabajo manual con comandos CLI avanzados"
    echo
    
    # Verificar argumentos de línea de comandos
    case "${1:-}" in
        --interactive|-i)
            interactive_setup
            ;;
        --help|-h)
            echo "Uso: $0 [opciones]"
            echo "Opciones:"
            echo "  --interactive, -i    Modo interactivo"
            echo "  --help, -h          Mostrar esta ayuda"
            echo "  --no-monitor        Deshabilitar monitoreo automático"
            echo
            echo "Configuración actual:"
            echo "  Directorio fuente: $SOURCE_DIR"
            echo "  Datasets: ${DATASETS[*]}"
            echo "  Perfil: $PROFILE"
            echo "  GPU: $GPU_ID"
            echo "  Comando: $(command -v autotrain &> /dev/null && echo "autotrain" || echo "python -m autotrain_sdk")"
            exit 0
            ;;
        --no-monitor)
            AUTO_MONITOR=false
            ;;
    esac
    
    # Ejecutar flujo de trabajo
    check_requirements
    import_datasets
    build_structure
    create_prompts
    generate_presets
    start_training
    monitor_training
    show_summary
}

# ====================================================================
# MANEJO DE ERRORES
# ====================================================================

# Capturar errores y limpiar
trap 'print_error "Script interrumpido o falló"' ERR

# Capturar Ctrl+C
trap 'echo; print_info "Script cancelado por usuario"; exit 0' INT

# ====================================================================
# EJECUCIÓN
# ====================================================================

main "$@" 
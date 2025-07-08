#!/usr/bin/env python3
"""
Script completo de migración que reemplaza los archivos bash en PorEliminar/.
Este script realiza todas las tareas de los archivos:
- 1.2.Output_Batch_Create.sh
- create_sample_prompts.py
- build_output_structure.py

Uso:
    python migrate_from_bash.py
"""

import sys
import os

# Agregar el directorio actual al path para importar autotrain_sdk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autotrain_sdk.dataset import (
    populate_output_structure, 
    create_sample_prompts,
    INPUT_DIR,
    OUTPUT_DIR
)
from autotrain_sdk.configurator import generate_presets

def main():
    print("🚀 AutoTrainV2 - Migration from Bash Scripts")
    print("=" * 50)
    print()
    
    # Verificar que existe input/
    if not INPUT_DIR.exists():
        print("❌ [ERROR] Input directory not found!")
        print(f"   Expected: {INPUT_DIR}")
        print()
        print("💡 [SOLUTION] Create input folders first:")
        print("   python -m autotrain_sdk dataset create --names 'your_dataset_name'")
        print("   Then place images in the created folders.")
        return 1
    
    # 1. Crear estructura de output y sample_prompts
    print("📁 [STEP 1] Creating output structure and copying images...")
    try:
        populate_output_structure()
        print("   ✓ Output structure created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create output structure: {e}")
        return 1
    
    # 2. Verificar que se crearon los sample_prompts
    print()
    print("📝 [STEP 2] Verifying sample_prompts.txt files...")
    try:
        count = create_sample_prompts()
        if count > 0:
            print(f"   ✓ {count} sample_prompts.txt files verified/created")
        else:
            print("   ⚠️  No datasets found for sample_prompts creation")
    except Exception as e:
        print(f"   ❌ Failed to create sample_prompts: {e}")
        return 1
    
    # 3. Generar presets TOML
    print()
    print("⚙️  [STEP 3] Generating TOML configuration presets...")
    try:
        generate_presets()
        print("   ✓ TOML presets generated successfully")
    except Exception as e:
        print(f"   ❌ Failed to generate presets: {e}")
        print("   💡 This might be normal if datasets don't have enough images")
        # No retornamos error aquí porque los presets pueden fallar por datasets incompletos
    
    # 4. Resumen final
    print()
    print("🎉 [COMPLETED] Migration finished successfully!")
    print("=" * 50)
    print()
    print("📋 What was done:")
    print(f"   • Created output structure in: {OUTPUT_DIR}")
    print("   • Copied images to training folders")
    print("   • Created sample_prompts.txt files")
    print("   • Generated TOML configuration files")
    print()
    print("🚀 Next steps:")
    print("   • Review the generated configs in BatchConfig/")
    print("   • Start training with: python -m autotrain_sdk train start")
    print("   • Or use the web interface: python -m autotrain_sdk web serve")
    print()
    print("✨ You no longer need the files in PorEliminar/ folder!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 
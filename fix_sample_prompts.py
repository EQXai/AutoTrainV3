#!/usr/bin/env python3
"""
Script para crear archivos sample_prompts.txt faltantes.
Esto soluciona el error al ejecutar combined_script.py cuando los archivos
sample_prompts.txt no existen.
"""

import sys
import os

# Agregar el directorio actual al path para importar autotrain_sdk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autotrain_sdk.dataset import create_sample_prompts

def main():
    print("[INFO] Creating missing sample_prompts.txt files...")
    print("=" * 50)
    
    try:
        count = create_sample_prompts()
        
        if count > 0:
            print(f"\n[SUCCESS] ✓ Created/updated {count} sample_prompts.txt files")
            print("\n[INFO] You can now run 'python combined_script.py' successfully!")
        else:
            print("\n[WARNING] ⚠️  No datasets found in output/ directory")
            print("\n[INFO] Please run output structure creation first:")
            print("  - From CLI: python -m autotrain_sdk dataset build-output")
            print("  - From Python: autotrain_sdk.populate_output_structure()")
            print("  - Or use the Gradio web interface")
            
    except Exception as e:
        print(f"\n[ERROR] ❌ Failed to create sample_prompts.txt files: {e}")
        print("\n[SOLUTION] Try running:")
        print("  python -m autotrain_sdk dataset create-prompts")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
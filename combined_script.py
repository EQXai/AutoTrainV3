import os
import toml
from pydantic import ValidationError
from autotrain_sdk.config_models import TrainingConfig

def verify_dataset_sample_prompts(project_root, dataset_name):
    """
    Verify that the dataset-specific sample prompts file exists in the output directory.
    The file should be created by the output structure creation script.
    
    Args:
        project_root (str): Path to the project root directory
        dataset_name (str): Name of the dataset
        
    Returns:
        str: Path to the sample prompts file if it exists, None otherwise
    """
    prompts_path = os.path.join(project_root, "output", dataset_name, "sample_prompts.txt")
    
    if os.path.exists(prompts_path):
        print(f"[INFO] Dataset sample prompts file found: {prompts_path}")
        return prompts_path
    else:
        print(f"[WARNING] Dataset sample prompts file not found: {prompts_path}")
        print(f"[WARNING] Please run the output structure creation script first (1.2.Output_Batch_Create.sh)")
        return None

def process_flux_checkpoint(project_root):
    print("[INFO] Starting FluxCheckpoint configuration...")
    
    # ------------------------------
    # Initial Setup for FluxCheckpoint
    # ------------------------------
    output_dir = os.path.join(project_root, "output")
    batch_config_dir = os.path.join(project_root, "BatchConfig", "Flux")
    os.makedirs(batch_config_dir, exist_ok=True)
    
    # ----------------------------------------------------------
    # Load base configuration (prefer external template if exists)
    # ----------------------------------------------------------
    template_path = os.path.join(project_root, "templates", "Flux", "base.toml")
    if os.path.isfile(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                config_dict_base_flux = toml.load(f)
        except toml.TomlDecodeError as e:
            print(f"[ERROR] Failed to parse FluxCheckpoint template '{template_path}': {e}")
            return False
    else:
        print(f"[ERROR] Template file '{template_path}' not found and no embedded configuration available.")
        return False

    # Get list of folders in output directory
    try:
        folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    except FileNotFoundError:
        print(f"[WARNING] The directory '{output_dir}' does not exist.")
        folders = []

    if not folders:
        print(f"[INFO] No folders found in '{output_dir}'.")
    else:
        for folder in folders:
            # Create necessary subdirectories
            subdirs = ["model", "log", "img"]
            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, folder, subdir)
                os.makedirs(subdir_path, exist_ok=True)

            # Create a copy of the base configuration
            config_dict = config_dict_base_flux.copy()

            # Verify dataset-specific sample prompts file exists
            dataset_prompts_path = verify_dataset_sample_prompts(project_root, folder)
            if dataset_prompts_path is None:
                print(f"[ERROR] Sample prompts file not found for dataset '{folder}'. Skipping...")
                continue

            # Replace placeholder in sample_prompts path
            if "sample_prompts" in config_dict:
                config_dict["sample_prompts"] = config_dict["sample_prompts"].replace("DATASET_OUTPUT_DIR", os.path.join(project_root, "output", folder))

            # Define keys that represent paths
            path_keys = [
                "ae",
                "clip_l",
                "pretrained_model_name_or_path",
                "t5xxl",
                "sample_prompts"
            ]

            # Convert relative paths to absolute paths with forward slashes
            for key in path_keys:
                if key in config_dict:
                    original_path = config_dict[key]
                    absolute_path = os.path.abspath(os.path.join(project_root, original_path))
                    absolute_path = absolute_path.replace('\\', '/')
                    config_dict[key] = absolute_path

            # Update specific paths based on folder name
            config_dict['output_dir'] = os.path.abspath(os.path.join(output_dir, folder, "model")).replace('\\', '/')
            config_dict['output_name'] = folder
            config_dict['logging_dir'] = os.path.abspath(os.path.join(output_dir, folder, "log")).replace('\\', '/')
            config_dict['train_data_dir'] = os.path.abspath(os.path.join(output_dir, folder, "img")).replace('\\', '/')

            # Verify the existence of paths
            required_paths = [
                config_dict.get("ae"),
                config_dict.get("clip_l"),
                config_dict.get("pretrained_model_name_or_path"),
                config_dict.get("t5xxl"),
                config_dict.get("sample_prompts"),
                config_dict.get("output_dir"),
                config_dict.get("logging_dir"),
                config_dict.get("train_data_dir")
            ]

            missing_paths = [path for path in required_paths if path and not os.path.exists(path)]
            if missing_paths:
                print(f"[WARNING] Some paths do not exist for folder '{folder}':")
                for path in missing_paths:
                    print(f"  - {path}")
                print("Please ensure that all necessary files and directories are present.\n")

            # ------------------------------------------------------
            # Validate config with Pydantic (ensures required fields)
            # ------------------------------------------------------
            try:
                TrainingConfig(**config_dict)
            except ValidationError as e:
                print(f"[ERROR] Validation failed for dataset '{folder}' (Flux):\n{e}\n")
                continue  # skip writing invalid config

            # Create the TOML file name and path
            toml_filename = f"{folder}.toml"
            toml_path = os.path.join(batch_config_dir, toml_filename)

            # Write the configuration in TOML format
            try:
                with open(toml_path, 'w') as toml_file:
                    toml.dump(config_dict, toml_file)
                print(f"[SUCCESS] FluxCheckpoint configuration file created: {toml_path}")
            except Exception as e:
                print(f"[ERROR] Failed to write FluxCheckpoint TOML file '{toml_path}': {e}")

    print("[INFO] FluxCheckpoint configuration completed.\n")
    return True

def process_flux_lora(project_root):
    print("[INFO] Starting FluxLora configuration...")
    
    # ------------------------------
    # Initial Setup for FluxLora
    # ------------------------------
    output_dir = os.path.join(project_root, "output")
    batch_config_dir = os.path.join(project_root, "BatchConfig", "FluxLORA")
    os.makedirs(batch_config_dir, exist_ok=True)
    
    # ----------------------------------------------------------
    # Load base configuration (prefer external template if exists)
    # ----------------------------------------------------------
    template_path = os.path.join(project_root, "templates", "FluxLORA", "base.toml")
    if os.path.isfile(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                config_dict_base_lora = toml.load(f)
        except toml.TomlDecodeError as e:
            print(f"[ERROR] Failed to parse FluxLora template '{template_path}': {e}")
            return False
    else:
        print(f"[ERROR] Template file '{template_path}' not found and no embedded configuration available.")
        return False

    # Get list of folders in output directory
    try:
        folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    except FileNotFoundError:
        print(f"[WARNING] The directory '{output_dir}' does not exist.")
        folders = []

    if not folders:
        print(f"[INFO] No folders found in '{output_dir}'.")
    else:
        for folder in folders:
            # Create necessary subdirectories
            subdirs = ["model", "log", "img"]
            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, folder, subdir)
                os.makedirs(subdir_path, exist_ok=True)

            # Create a copy of the base configuration
            config_dict = config_dict_base_lora.copy()

            # Verify dataset-specific sample prompts file exists
            dataset_prompts_path = verify_dataset_sample_prompts(project_root, folder)
            if dataset_prompts_path is None:
                print(f"[ERROR] Sample prompts file not found for dataset '{folder}'. Skipping...")
                continue

            # Replace placeholder in sample_prompts path
            if "sample_prompts" in config_dict:
                config_dict["sample_prompts"] = config_dict["sample_prompts"].replace("DATASET_OUTPUT_DIR", os.path.join(project_root, "output", folder))

            # Define keys that represent paths
            path_keys = [
                "ae",
                "clip_l",
                "pretrained_model_name_or_path",
                "t5xxl",
                "sample_prompts",
                "output_dir",
                "logging_dir",
                "train_data_dir"
            ]

            # Convert relative paths to absolute paths with forward slashes
            for key in path_keys:
                if key in config_dict and config_dict[key] != "X":
                    original_path = config_dict[key]
                    absolute_path = os.path.abspath(os.path.join(project_root, original_path))
                    absolute_path = absolute_path.replace('\\', '/')
                    config_dict[key] = absolute_path
                elif key in config_dict and config_dict[key] == "X":
                    # Update paths based on the current folder
                    if key == "output_dir":
                        config_dict[key] = os.path.abspath(os.path.join(output_dir, folder, "model")).replace('\\', '/')
                    elif key == "logging_dir":
                        config_dict[key] = os.path.abspath(os.path.join(output_dir, folder, "log")).replace('\\', '/')
                    elif key == "train_data_dir":
                        config_dict[key] = os.path.abspath(os.path.join(output_dir, folder, "img")).replace('\\', '/')
                    else:
                        # For keys that are set to "X" and not explicitly handled
                        config_dict[key] = "X"

            # Update 'output_name' based on folder name
            config_dict['output_name'] = folder

            # Verify the existence of paths
            required_paths = [
                config_dict.get("ae"),
                config_dict.get("clip_l"),
                config_dict.get("pretrained_model_name_or_path"),
                config_dict.get("t5xxl"),
                config_dict.get("sample_prompts"),
                config_dict.get("output_dir"),
                config_dict.get("logging_dir"),
                config_dict.get("train_data_dir")
            ]

            missing_paths = [path for path in required_paths if path and path != "X" and not os.path.exists(path)]
            if missing_paths:
                print(f"[WARNING] Some paths do not exist for folder '{folder}':")
                for path in missing_paths:
                    print(f"  - {path}")
                print("Please ensure that all necessary files and directories are present.\n")
            else:
                print(f"[INFO] All required paths exist for folder '{folder}'.")

            # ------------------------------------------------------
            # Validate config with Pydantic
            # ------------------------------------------------------
            try:
                TrainingConfig(**config_dict)
            except ValidationError as e:
                print(f"[ERROR] Validation failed for dataset '{folder}' (FluxLORA):\n{e}\n")
                continue

            # Create the TOML file name and path
            toml_filename = f"{folder}.toml"
            toml_path = os.path.join(batch_config_dir, toml_filename)

            # Write the configuration in TOML format
            try:
                with open(toml_path, 'w') as toml_file:
                    toml.dump(config_dict, toml_file)
                print(f"[SUCCESS] FluxLora configuration file created: {toml_path}")
            except Exception as e:
                print(f"[ERROR] Failed to write FluxLora TOML file '{toml_path}': {e}")

    print("[INFO] FluxLora configuration completed.\n")
    return True

def process_sdxl_nude(project_root):
    print("[INFO] Starting SDXLNude configuration...")
    
    # ------------------------------
    # Initial Setup for SDXLNude
    # ------------------------------
    output_dir = os.path.join(project_root, "output")
    batch_config_dir = os.path.join(project_root, "BatchConfig", "Nude")
    os.makedirs(batch_config_dir, exist_ok=True)
    
    # ----------------------------------------------------------
    # Load base configuration (prefer external template if exists)
    # ----------------------------------------------------------
    template_path = os.path.join(project_root, "templates", "Nude", "base.toml")
    if os.path.isfile(template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                config_dict_base_nude = toml.load(f)
        except toml.TomlDecodeError as e:
            print(f"[ERROR] Failed to parse SDXLNude template '{template_path}': {e}")
            return False
    else:
        print(f"[ERROR] Template file '{template_path}' not found and no embedded configuration available.")
        return False

    # Get list of folders in output directory
    try:
        folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    except FileNotFoundError:
        print(f"[WARNING] The directory '{output_dir}' does not exist.")
        folders = []

    if not folders:
        print(f"[INFO] No folders found in '{output_dir}'.")
    else:
        for folder in folders:
            # Create necessary subdirectories
            subdirs = ["model", "log", "img"]
            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, folder, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                print(f"[INFO] Subdirectory '{subdir_path}' ensured.")

            # Create a copy of the base configuration
            config_dict = config_dict_base_nude.copy()

            # Verify dataset-specific sample prompts file exists
            dataset_prompts_path = verify_dataset_sample_prompts(project_root, folder)
            if dataset_prompts_path is None:
                print(f"[ERROR] Sample prompts file not found for dataset '{folder}'. Skipping...")
                continue

            # Replace placeholder in sample_prompts path
            if "sample_prompts" in config_dict:
                config_dict["sample_prompts"] = config_dict["sample_prompts"].replace("DATASET_OUTPUT_DIR", os.path.join(project_root, "output", folder))

            # Define keys that represent paths
            path_keys = [
                "pretrained_model_name_or_path",
                "sample_prompts"
            ]

            # Convert relative paths to absolute paths with forward slashes
            for key in path_keys:
                if key in config_dict:
                    if key == "pretrained_model_name_or_path":
                        # Special handling for pretrained model path
                        absolute_path = os.path.abspath(os.path.join(project_root, "models/trainX/NudeSDXLModel.safetensors"))
                        absolute_path = absolute_path.replace('\\', '/')
                        config_dict[key] = absolute_path
                    else:
                        # General path handling
                        original_path = config_dict[key]
                        absolute_path = os.path.abspath(os.path.join(project_root, original_path))
                        absolute_path = absolute_path.replace('\\', '/')
                        config_dict[key] = absolute_path

            # Update specific paths based on folder name
            config_dict['output_dir'] = os.path.abspath(os.path.join(output_dir, folder, "model")).replace('\\', '/')
            config_dict['output_name'] = folder
            config_dict['logging_dir'] = os.path.abspath(os.path.join(output_dir, folder, "log")).replace('\\', '/')
            config_dict['train_data_dir'] = os.path.abspath(os.path.join(output_dir, folder, "img")).replace('\\', '/')

            # Verify the existence of paths
            required_paths = [
                config_dict.get("pretrained_model_name_or_path"),
                config_dict.get("sample_prompts"),
                config_dict.get("output_dir"),
                config_dict.get("logging_dir"),
                config_dict.get("train_data_dir")
            ]

            missing_paths = [path for path in required_paths if path and not os.path.exists(path)]
            if missing_paths:
                print(f"[WARNING] Some paths do not exist for folder '{folder}':")
                for path in missing_paths:
                    print(f"  - {path}")
                print("Please ensure that all necessary files and directories are present.\n")
            else:
                print(f"[INFO] All required paths exist for folder '{folder}'.")

            # ------------------------------------------------------
            # Validate config with Pydantic
            # ------------------------------------------------------
            try:
                TrainingConfig(**config_dict)
            except ValidationError as e:
                print(f"[ERROR] Validation failed for dataset '{folder}' (Nude):\n{e}\n")
                continue

            # Create the TOML file name and path
            toml_filename = f"{folder}.toml"
            toml_path = os.path.join(batch_config_dir, toml_filename)

            # Write the configuration in TOML format
            try:
                with open(toml_path, 'w') as toml_file:
                    toml.dump(config_dict, toml_file)
                print(f"[SUCCESS] SDXLNude configuration file created: {toml_path}\n")
            except Exception as e:
                print(f"[ERROR] Failed to write SDXLNude TOML file '{toml_path}': {e}\n")

        print("[INFO] All SDXLNude configuration files have been generated successfully.\n")

    # Verify required subdirectories
    if folders:
        all_missing = False
        for folder in folders:
            required_subdirs = [
                os.path.join(output_dir, folder, "model"),
                os.path.join(output_dir, folder, "log"),
                os.path.join(output_dir, folder, "img")
            ]

            for path in required_subdirs:
                if not os.path.exists(path):
                    print(f"[ERROR] Required subdirectory '{path}' is missing.")
                    all_missing = True

        if all_missing:
            print("\n[ERROR] Some required directories are missing. Please ensure all directories are correctly set up.")
        else:
            print("\n[INFO] All required directories are present.")

    print("[INFO] SDXLNude configuration completed.\n")
    return True

def main():
    # Path to the project's root directory (where this script is located)
    project_root = os.path.abspath(os.path.dirname(__file__))

    # Execute FluxCheckpoint configuration
    flux_checkpoint_success = process_flux_checkpoint(project_root)
    if flux_checkpoint_success:
        print("[SUCCESS] FluxCheckpoint executed successfully.\n")
    else:
        print("[FAILURE] FluxCheckpoint encountered errors.\n")

    # Execute FluxLora configuration
    flux_lora_success = process_flux_lora(project_root)
    if flux_lora_success:
        print("[SUCCESS] FluxLora executed successfully.\n")
    else:
        print("[FAILURE] FluxLora encountered errors.\n")

    # Execute SDXLNude configuration
    sdxl_nude_success = process_sdxl_nude(project_root)
    if sdxl_nude_success:
        print("[SUCCESS] SDXLNude executed successfully.\n")
    else:
        print("[FAILURE] SDXLNude encountered errors.\n")

    # Final status
    if flux_checkpoint_success and flux_lora_success and sdxl_nude_success:
        print("[INFO] All configurations have been executed successfully.")
    else:
        print("[INFO] Some configurations encountered errors. Please check the logs above.")

if __name__ == "__main__":
    main()

# ultimate_morph_generator/main_orchestrator.py
import argparse
import os
import sys
import random
import numpy as np
import torch
import logging

# Ensure the package root is in PYTHONPATH if running as a script directly for dev
# This is usually not needed if running with `python -m ultimate_morph_generator.main_orchestrator`
# current_dir = os.path.dirname(os.path.abspath(__file__))
# package_root = os.path.dirname(current_dir) # Assuming main_orchestrator.py is one level down from package root
# if package_root not in sys.path:
#    sys.path.insert(0, package_root)

from .config import get_config, SystemConfig
from .utilities.logging_config import setup_logging
from .core_logic.generation_manager import GenerationManager

# --- Global Logger Setup (initialized once) ---
# Logger will be configured based on the loaded SystemConfig.
# We call setup_logging() after config is loaded.
logger = None


def print_welcome_message(config: SystemConfig):
    """Prints a welcome message with key configuration details."""
    if logger:  # Check if logger is initialized
        logger.info("=======================================================")
        logger.info(f"  Starting: {config.project_name} v{config.version}")
        logger.info(f"  Target Character: '{config.target_character_string}' (Index: {config.target_character_index})")
        logger.info(f"  Output Directory: {config.data_management.output_base_dir}")
        logger.info(f"  Device: {config.get_actual_device()}")
        logger.info("=======================================================")
    else:  # Fallback to print if logger setup failed or called too early
        print("Welcome to the Ultimate Morph Generator!")
        print(f"Target: {config.target_character_string}, Output: {config.data_management.output_base_dir}")


def setup_environment(config: SystemConfig):
    """Sets up the environment, e.g., random seeds."""
    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
            # Potentially set deterministic algorithms, but can impact performance
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logger.info(f"Global random seed set to: {config.random_seed}")

    # Create base output directory if it doesn't exist
    os.makedirs(config.data_management.output_base_dir, exist_ok=True)
    logger.info(f"Ensured output base directory exists: {config.data_management.output_base_dir}")


def main():
    """
    Main function to parse arguments, load configuration, set up components,
    and start the generation process.
    """
    global logger  # Allow main to assign to the global logger variable

    parser = argparse.ArgumentParser(description="Ultimate Morphological Character Generator.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,  # Default to None, get_config will use internal defaults if not provided
        help="Path to a custom YAML configuration file."
    )
    # Add other command-line arguments if needed to override specific config values
    # e.g., --target-char, --max-library-size, etc.
    # These would then update the loaded config object.

    args = parser.parse_args()

    # 1. Load Configuration
    # get_config is a singleton, will load from file if path provided, else defaults.
    try:
        system_config = get_config(config_file_path=args.config)
        if args.config and not os.path.exists(args.config):
            print(f"Warning: Specified config file '{args.config}' not found. Using default configuration.")
        elif args.config:
            print(f"Successfully loaded configuration from: {args.config}")
        else:
            print("No custom config file specified. Using default configuration.")
    except Exception as e:
        print(f"CRITICAL: Failed to load or initialize configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Setup Logging (now that config is loaded)
    # The logger from logging_config is typically named after project_name.
    # We get an instance of it here.
    try:
        logger = setup_logging()  # Uses the global config instance internally
    except Exception as e:
        print(f"CRITICAL: Failed to setup logging: {e}", file=sys.stderr)
        # Continue without file logging if console logging works, or exit
        logger = logging.getLogger(system_config.project_name)  # Basic fallback logger
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel("INFO")
        logger.warning("Logging setup failed, using basic console logger.")

    # 3. Print Welcome Message and Setup Environment
    print_welcome_message(system_config)
    setup_environment(system_config)

    # 4. Initialize the GenerationManager
    # This is where all major components are instantiated and wired together.
    # The GenerationManager's __init__ handles this complex setup.
    try:
        logger.info("Initializing Generation Manager...")
        generation_manager = GenerationManager(cfg_system=system_config)  # Pass the loaded config
        logger.info("Generation Manager initialized successfully.")
    except ValueError as ve:  # Catch specific errors like "No initial seed samples"
        logger.critical(f"Failed to initialize GenerationManager: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during GenerationManager initialization: {e}",
                        exc_info=True)
        sys.exit(1)

    # 5. Run the Generation Loop
    try:
        logger.info("Starting the morphological generation process...")
        generation_manager.run_generation_loop()
        logger.info("Morphological generation process finished.")
    except KeyboardInterrupt:
        logger.warning("Generation process interrupted by user (Ctrl+C).")
        # Perform any necessary cleanup if model was training, etc.
    except Exception as e:
        logger.error(f"An critical error occurred during the generation loop: {e}", exc_info=True)
        # Potentially save partial state or library if possible
    finally:
        logger.info("Shutting down the Ultimate Morph Generator.")
        # Any final cleanup tasks can go here.
        # E.g., closing database connections if not handled by context managers/destructors.


if __name__ == "__main__":
    # This allows the script to be run directly (e.g., `python main_orchestrator.py`)
    # or as part of a package (`python -m ultimate_morph_generator.main_orchestrator`)
    main()
"""
A runnable script to perform the expensive pre-computation of agent policies.

This script uses the configuration from a specified YAML file to generate and save
a policy library, which can then be loaded by the main simulation model.
"""
import argparse
import logging

from dgl_ptm.config import SVEIRConfig
from dgl_ptm.model.policy_precomputation import generate_and_save_policy_library

# Setup basic logging to see the progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to parse arguments and trigger policy generation.
    """
    try:
        # Load the configuration object from the YAML file
        config = SVEIRConfig()
    except Exception as e:
        print(f"\nError: Could not load or parse the configuration file.")
        print(f"Details: {e}")
        return

    print("Configuration loaded. Starting policy pre-computation... (This may take a while)")
    
    # Trigger the function that does the heavy lifting
    generate_and_save_policy_library(config)
    
    print("\nPolicy library successfully generated and saved.")
    print(f"File saved at: {config.policy_library_path}")


if __name__ == "__main__":
    main()
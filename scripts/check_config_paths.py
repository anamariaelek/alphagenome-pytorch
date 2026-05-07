#!/usr/bin/env python3
"""
Check if all paths in a finetuning config file exist.

Usage:
    python scripts/check_config_paths.py configs/finetune_all_132kb.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import yaml


def expand_path(path_str):
    """Expand environment variables and ~ in a path string."""
    if not isinstance(path_str, str):
        return path_str
    # Expand environment variables
    expanded = os.path.expandvars(path_str)
    # Expand ~ for home directory
    expanded = os.path.expanduser(expanded)
    return expanded


def check_path_exists(path_str, path_type="file"):
    """Check if a path exists and return status."""
    if not isinstance(path_str, str):
        return None, "Not a path string"
    
    expanded = expand_path(path_str)
    path = Path(expanded)
    
    if path.exists():
        if path_type == "file" and not path.is_file():
            return False, f"Exists but not a file: {expanded}"
        elif path_type == "dir" and not path.is_dir():
            return False, f"Exists but not a directory: {expanded}"
        return True, expanded
    else:
        return False, f"Does not exist: {expanded}"


def check_config_paths(config_path):
    """Check all paths in the config file."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Checking paths in: {config_path}\n")
    
    all_valid = True
    
    # Check species-specific paths
    if 'species' in config:
        print("=" * 70)
        print("SPECIES-SPECIFIC PATHS")
        print("=" * 70)
        
        for species in config['species']:
            species_name = species.get('name', 'unknown')
            print(f"\n{species_name.upper()}:")
            print("-" * 70)
            
            # Check each path type
            path_keys = ['genome', 'annotation_parquet', 'usage_parquet', 
                        'train_bed', 'val_bed']
            
            for key in path_keys:
                if key in species:
                    exists, msg = check_path_exists(species[key], path_type="file")
                    status = "✓" if exists else "✗"
                    color = "\033[92m" if exists else "\033[91m"
                    reset = "\033[0m"
                    
                    print(f"  {status} {key:20s} {color}{msg}{reset}")
                    
                    if not exists:
                        all_valid = False
    
    # Check pretrained weights
    if 'pretrained_weights' in config:
        print(f"\n{'=' * 70}")
        print("PRETRAINED WEIGHTS")
        print("=" * 70)
        
        exists, msg = check_path_exists(config['pretrained_weights'], path_type="file")
        status = "✓" if exists else "✗"
        color = "\033[92m" if exists else "\033[91m"
        reset = "\033[0m"
        
        print(f"  {status} pretrained_weights  {color}{msg}{reset}")
        
        if not exists:
            all_valid = False
    
    # Check output directory (should exist or be creatable)
    if 'output_dir' in config:
        print(f"\n{'=' * 70}")
        print("OUTPUT DIRECTORY")
        print("=" * 70)
        
        expanded = expand_path(config['output_dir'])
        path = Path(expanded)
        
        if path.exists():
            if path.is_dir():
                print(f"  ✓ output_dir         \033[92mExists: {expanded}\033[0m")
            else:
                print(f"  ✗ output_dir         \033[91mExists but not a directory: {expanded}\033[0m")
                all_valid = False
        else:
            # Check if parent exists and is writable
            parent = path.parent
            if parent.exists() and os.access(parent, os.W_OK):
                print(f"  ⚠ output_dir         \033[93mWill be created: {expanded}\033[0m")
            else:
                print(f"  ✗ output_dir         \033[91mParent doesn't exist or not writable: {expanded}\033[0m")
                all_valid = False
    
    # Summary
    print(f"\n{'=' * 70}")
    if all_valid:
        print("\033[92m✓ All paths are valid!\033[0m")
        return 0
    else:
        print("\033[91m✗ Some paths are missing or invalid!\033[0m")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Check if all paths in a finetuning config exist"
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to config YAML file'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1
    
    return check_config_paths(args.config)


if __name__ == '__main__':
    sys.exit(main())

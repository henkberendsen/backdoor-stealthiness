#!/usr/bin/env python3
"""
Script to automatically fix hardcoded paths in all BackdoorBench attack results.
This script iterates through all attack directories in large_files/record/ and applies
the path fix for attacks that were trained using BackdoorBench.

The script automatically detects the old hardcoded path from each attack_result.pt file,
so it works even if different attacks were created on different machines.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

# Define the BackdoorBench attacks (from eval.ipynb)
BACKDOORBENCH_ATTACKS = ["badnet", "blended", "wanet", "bpp", "narcissus"]

# Get the absolute path to the repository root
REPO_ROOT = Path(__file__).parent.resolve()

# Define paths
RECORD_DIR = REPO_ROOT / "large_files" / "record"
CHANGE_PATHS_SCRIPT = REPO_ROOT / "change_hardcoded_paths.py"

# New path (current machine)
NEW_PATH = str(RECORD_DIR) + "/"

def is_backdoorbench_attack(dir_name):
    """Check if a directory name corresponds to a BackdoorBench attack."""
    for attack in BACKDOORBENCH_ATTACKS:
        if dir_name.startswith(attack + "_"):
            return True
    return False


def detect_old_path(attack_result_file):
    """
    Automatically detect the old hardcoded path from an attack_result.pt file.
    
    Args:
        attack_result_file (Path): Path to the attack_result.pt file
        
    Returns:
        str or None: The detected old path, or None if detection failed
    """
    try:
        # Load the attack result
        atk_dict = torch.load(attack_result_file, weights_only=False)
        
        # Try to find a path in the data structure
        for key in ['bd_train', 'bd_test', 'cross_test']:
            if key not in atk_dict:
                continue
                
            try:
                # Try to get save_folder_path
                save_folder_path = atk_dict[key].get('save_folder_path', None)
                if save_folder_path and isinstance(save_folder_path, str):
                    # Extract the base path (everything before and including 'record/')
                    if 'record/' in save_folder_path or 'record\\' in save_folder_path:
                        # Find the position of 'record/'
                        idx = save_folder_path.find('record/')
                        if idx == -1:
                            idx = save_folder_path.find('record\\')
                        
                        if idx != -1:
                            # Extract path up to and including 'record/'
                            old_path = save_folder_path[:idx + len('record/')]
                            return old_path
                
                # Try to get path from bd_data_container
                bd_data_container = atk_dict[key].get('bd_data_container', None)
                if bd_data_container:
                    data_dict = bd_data_container.get('data_dict', {})
                    
                    # Get first available path from data_dict
                    for idx in data_dict.keys():
                        path = data_dict[idx].get('path', None)
                        if path and isinstance(path, str):
                            # Extract base path
                            if 'record/' in path or 'record\\' in path:
                                idx_pos = path.find('record/')
                                if idx_pos == -1:
                                    idx_pos = path.find('record\\')
                                
                                if idx_pos != -1:
                                    old_path = path[:idx_pos + len('record/')]
                                    return old_path
                        break  # Only check first entry
                        
            except Exception:
                continue
                
    except Exception as e:
        print(f"    Warning: Could not detect old path - {e}")
    
    return None

def fix_attack_paths(attack_dir):
    """Fix hardcoded paths for a single attack directory."""
    attack_result_file = attack_dir / "attack_result.pt"
    
    # Check if attack_result.pt exists
    if not attack_result_file.exists():
        print(f"  ⚠️  Skipping {attack_dir.name}: attack_result.pt not found")
        return False
    
    # Detect the old path from the file itself
    print(f"  🔍 Detecting old hardcoded path...")
    old_path = detect_old_path(attack_result_file)
    
    if old_path is None:
        print(f"  ⚠️  Could not detect old path in {attack_dir.name}")
        print(f"      Skipping this directory")
        return False
    
    print(f"  📁 Detected old path: {old_path}")
    print(f"  📁 New path: {NEW_PATH}")
    
    # Run the change_hardcoded_paths.py script
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(CHANGE_PATHS_SCRIPT),
        "--old", old_path,
        "--new", NEW_PATH,
        "--attack_result", str(attack_result_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Successfully fixed paths in {attack_dir.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error fixing {attack_dir.name}:")
        print(f"    {e.stderr}")
        return False

def main():
    """Main function to process all BackdoorBench attacks."""
    print("=" * 70)
    print("BackdoorBench Attack Path Fixer (Auto-Detect Mode)")
    print("=" * 70)
    print(f"\nRepository root: {REPO_ROOT}")
    print(f"Record directory: {RECORD_DIR}")
    print(f"New path: {NEW_PATH}")
    print(f"\nBackdoorBench attacks to fix: {', '.join(BACKDOORBENCH_ATTACKS)}")
    print(f"\nNote: Old paths will be automatically detected from each attack_result.pt file")
    print("=" * 70)
    
    # Check if record directory exists
    if not RECORD_DIR.exists():
        print(f"\n❌ Error: Record directory not found at {RECORD_DIR}")
        sys.exit(1)
    
    # Check if change_hardcoded_paths.py exists
    if not CHANGE_PATHS_SCRIPT.exists():
        print(f"\n❌ Error: change_hardcoded_paths.py not found at {CHANGE_PATHS_SCRIPT}")
        sys.exit(1)
    
    # Get all directories in record/
    attack_dirs = [d for d in RECORD_DIR.iterdir() if d.is_dir()]
    
    # Filter for BackdoorBench attacks
    backdoorbench_dirs = [d for d in attack_dirs if is_backdoorbench_attack(d.name)]
    
    if not backdoorbench_dirs:
        print("\n⚠️  No BackdoorBench attack directories found!")
        return
    
    print(f"\nFound {len(backdoorbench_dirs)} BackdoorBench attack directories to process:\n")
    
    # Process each BackdoorBench attack directory
    success_count = 0
    fail_count = 0
    
    for i, attack_dir in enumerate(sorted(backdoorbench_dirs), 1):
        print(f"[{i}/{len(backdoorbench_dirs)}] Processing: {attack_dir.name}")
        
        if fix_attack_paths(attack_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total processed: {len(backdoorbench_dirs)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print("=" * 70)
    
    if fail_count == 0:
        print("\n✅ All BackdoorBench attack paths have been successfully fixed!")
    else:
        print(f"\n⚠️  {fail_count} attack(s) failed. Please check the errors above.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to remove Neptune tags from configuration files
"""

import os
import glob

def remove_neptune_tags_from_file(file_path):
    """Remove neptune_tags line from a configuration file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find and remove neptune_tags line
        new_lines = []
        removed = False
        for line in lines:
            if line.strip().startswith('neptune_tags ='):
                removed = True
                print(f"Removed neptune_tags from {file_path}")
                continue
            new_lines.append(line)
        
        if removed:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Remove Neptune tags from all configuration files."""
    # Find all Python configuration files
    config_patterns = [
        "configs_reid/**/*.py",
        "configs/**/*.py"
    ]
    
    files_processed = 0
    files_modified = 0
    
    for pattern in config_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            files_processed += 1
            if remove_neptune_tags_from_file(file_path):
                files_modified += 1
    
    print(f"\nSummary:")
    print(f"Files processed: {files_processed}")
    print(f"Files modified: {files_modified}")

if __name__ == "__main__":
    main()
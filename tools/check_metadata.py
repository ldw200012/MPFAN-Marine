#!/usr/bin/env python3
"""
Script to inspect the metadata.pkl file and see what objects are included.
"""

import pickle
import os

def check_metadata(metadata_path):
    """Check what objects are in the metadata file."""
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print("=== METADATA INSPECTION ===")
    print(f"Total objects: {len(metadata['obj_infos'])}")
    print(f"Total frames: {len(metadata['frame_infos'])}")
    print(f"Scenes: {len(metadata['scene_infos'])}")
    
    print("\n=== OBJECTS IN METADATA ===")
    all_targets = list(metadata['obj_infos'].keys())
    
    # Separate regular targets from FP targets
    regular_targets = [t for t in all_targets if not t.startswith('FP')]
    fp_targets = [t for t in all_targets if t.startswith('FP')]
    
    # Sort regular targets numerically (target1, target2, etc.)
    regular_targets.sort(key=lambda x: int(x.replace('target', '')))
    
    print("Regular targets:")
    for i, obj_id in enumerate(regular_targets):
        obj_info = metadata['obj_infos'][obj_id]
        frame_count = len(obj_info['num_pts'])
        print(f"  {i+1:2d}. {obj_id}: {frame_count} frames")
    
    print("\nFP targets:")
    for i, obj_id in enumerate(fp_targets):
        obj_info = metadata['obj_infos'][obj_id]
        frame_count = len(obj_info['num_pts'])
        print(f"  {i+1:2d}. {obj_id}: {frame_count} frames")
    
    print("\n=== TRAIN/VAL SPLIT ANALYSIS ===")
    # Apply the same split logic as in load_jeongok_metadata
    train_targets = regular_targets[:6]  # target1-target6
    val_targets = regular_targets[6:8]   # target7-target8
    
    print(f"Regular targets: {regular_targets}")
    print(f"FP targets: {fp_targets}")
    print(f"Train targets (first 6): {train_targets}")
    print(f"Val targets (last 2): {val_targets}")
    
    print(f"\nExpected train objects: {train_targets + fp_targets}")
    print(f"Expected val objects: {val_targets + fp_targets}")

if __name__ == "__main__":
    metadata_path = "Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/metadata/metadata.pkl"
    check_metadata(metadata_path) 
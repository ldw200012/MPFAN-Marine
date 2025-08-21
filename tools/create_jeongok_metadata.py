#!/usr/bin/env python3
"""
Script to create metadata.pkl file for Jeongok-ReID dataset.
This script scans the dataset directory and creates the necessary metadata structure.
"""

import os
import pickle
import numpy as np
from pathlib import Path
import argparse

def create_jeongok_metadata(data_root, output_path):
    """
    Create metadata.pkl file for Jeongok-ReID dataset.
    
    Args:
        data_root: Path to the dataset root directory
        output_path: Path where to save the metadata.pkl file
    """
    
    # Create metadata structure
    metadata = {
        'scene_infos': {},
        'obj_infos': {},
        'frame_infos': {}
    }
    
    # Base path for objects
    objects_path = os.path.join(data_root, 'objects')
    
    if not os.path.exists(objects_path):
        print(f"Error: Objects directory not found at {objects_path}")
        return False
    
    print(f"Scanning objects directory: {objects_path}")
    
    # Scan all object directories
    for obj_id in os.listdir(objects_path):
        obj_path = os.path.join(objects_path, obj_id)
        
        if not os.path.isdir(obj_path):
            continue
            
        print(f"Processing object: {obj_id}")
        
        # Initialize object info
        obj_info = {
            'id': obj_id,
            'class_name': 'FP_boat' if obj_id.startswith('FP') else 'boat',
            'num_pts': {},
            'visibility': {},
            'path': f'{obj_id}',
            'scene_id': 'jeongok_scene',  # Single scene for Jeongok
            'tracking_id': obj_id
        }
        
        # Scan frame directories
        frame_count = 0
        for frame_dir in os.listdir(obj_path):
            frame_path = os.path.join(obj_path, frame_dir)
            
            if not os.path.isdir(frame_path):
                continue
                
            try:
                frame_id = int(frame_dir)
            except ValueError:
                print(f"Warning: Skipping non-numeric frame directory: {frame_dir}")
                continue
            
            # Count points in this frame
            point_count = 0
            txt_files = [f for f in os.listdir(frame_path) if f.endswith('.txt')]
            
            if txt_files:
                # Use the first txt file to get point count
                txt_file = txt_files[0]
                try:
                    point_count = int(txt_file.split('.')[0])
                except ValueError:
                    print(f"Warning: Could not parse point count from {txt_file}")
                    point_count = 0
            
            if point_count > 0:
                obj_info['num_pts'][frame_id] = point_count
                
                # Calculate visibility based on point count (similar to original logic)
                # We'll need to find max points across all frames first
                frame_count += 1
        
        # Calculate visibility for each frame
        if obj_info['num_pts']:
            max_points = max(obj_info['num_pts'].values())
            
            for frame_id, point_count in obj_info['num_pts'].items():
                # Split into Q4 visibility levels
                if 0 <= point_count <= max_points * 0.4:
                    obj_info['visibility'][frame_id] = '1'
                elif max_points * 0.4 < point_count <= max_points * 0.6:
                    obj_info['visibility'][frame_id] = '2'
                elif max_points * 0.6 < point_count <= max_points * 0.8:
                    obj_info['visibility'][frame_id] = '3'
                elif max_points * 0.8 < point_count <= max_points:
                    obj_info['visibility'][frame_id] = '4'
                else:
                    obj_info['visibility'][frame_id] = '1'
        
        # Add object info to metadata
        metadata['obj_infos'][obj_id] = obj_info
        
        # Add frame info
        for frame_id in obj_info['num_pts'].keys():
            frame_key = f"{obj_id}_{frame_id}"
            metadata['frame_infos'][frame_key] = {
                'obj_id': obj_id,
                'frame_id': frame_id,
                'scene_id': 'jeongok_scene',
                'num_points': obj_info['num_pts'][frame_id],
                'visibility': obj_info['visibility'][frame_id]
            }
    
    # Add scene info
    metadata['scene_infos']['jeongok_scene'] = {
        'id': 'jeongok_scene',
        'name': 'Jeongok Port Scene',
        'description': 'Jeongok Port LiDAR dataset for ReID'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save metadata
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nMetadata created successfully!")
    print(f"Output file: {output_path}")
    print(f"Total objects: {len(metadata['obj_infos'])}")
    print(f"Total frames: {len(metadata['frame_infos'])}")
    print(f"Scenes: {len(metadata['scene_infos'])}")
    
    # Print object summary
    print(f"\nObject summary:")
    for obj_id, obj_info in metadata['obj_infos'].items():
        frame_count = len(obj_info['num_pts'])
        print(f"  {obj_id}: {frame_count} frames")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Create metadata.pkl for Jeongok-ReID dataset')
    parser.add_argument('--data-root', type=str, 
                       default='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both',
                       help='Path to dataset root directory')
    parser.add_argument('--output-path', type=str,
                       default='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/metadata/metadata.pkl',
                       help='Path to output metadata.pkl file')
    
    args = parser.parse_args()
    
    print("Creating Jeongok-ReID metadata...")
    print(f"Data root: {args.data_root}")
    print(f"Output path: {args.output_path}")
    
    success = create_jeongok_metadata(args.data_root, args.output_path)
    
    if success:
        print("\n✅ Metadata creation completed successfully!")
    else:
        print("\n❌ Metadata creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
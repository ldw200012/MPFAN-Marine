#!/usr/bin/env python3
"""
Script to generate precomputed positive and negative pairs for Jeongok-ReID dataset and save as PKL.
"""
import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# --- Helper: load_jeongok_metadata (copied from object_loader_base.py) ---
def load_jeongok_metadata(metadata_path, train):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    all_targets = list(metadata['obj_infos'].keys())

    print(all_targets)

    regular_targets = [t for t in all_targets if not t.startswith('FP')]
    fp_targets = [t for t in all_targets if t.startswith('FP')]
    regular_targets.sort(key=lambda x: int(x.replace('target', '')))
    train_targets = regular_targets[:10]
    val_targets = regular_targets[10:13]
    if train:
        valid_targets = train_targets + fp_targets
    else:
        valid_targets = val_targets + fp_targets
    filtered_metadata = {
        'scene_infos': metadata['scene_infos'],
        'obj_infos': {k: v for k, v in metadata['obj_infos'].items() if k in valid_targets},
        'frame_infos': {k: v for k, v in metadata['frame_infos'].items() if any(target in k for target in valid_targets)}
    }
    return filtered_metadata, valid_targets

# --- Main pair generation logic ---
def generate_pairs(metadata, valid_targets, K=200, seed=42):
    np.random.seed(seed)
    obj_infos = metadata['obj_infos']
    # Assign class indices: vessels first, then FP
    vessel_targets = [t for t in valid_targets if not t.startswith('FP')]
    fp_targets = [t for t in valid_targets if t.startswith('FP')]
    class_map = {t: i for i, t in enumerate(vessel_targets + fp_targets)}
    all_targets = vessel_targets + fp_targets
    pairs = []
    for obj in tqdm(all_targets, desc='Classes'):
        frames = list(obj_infos[obj]['num_pts'].keys())
        if len(frames) < 2:
            continue
        # --- Positive pairs ---
        all_pos = []
        for i in range(len(frames)):
            for j in range(i+1, len(frames)):
                all_pos.append((frames[i], frames[j]))
        if len(all_pos) > K:
            pos_pairs = list(np.random.choice(len(all_pos), K, replace=False))
            pos_pairs = [all_pos[i] for i in pos_pairs]
        else:
            pos_pairs = all_pos
        for f1, f2 in pos_pairs:
            pairs.append({
                'type': 'pos',
                'anchor_obj': obj,
                'anchor_frame': f1,
                'pair_obj': obj,
                'pair_frame': f2,
                'anchor_class': class_map[obj],
                'pair_class': class_map[obj],
            })
        # --- Negative pairs (one per positive) ---
        for f1, _ in pos_pairs:
            # Pick a random other class
            other_classes = [t for t in all_targets if t != obj]
            neg_obj = np.random.choice(other_classes)
            neg_frames = list(obj_infos[neg_obj]['num_pts'].keys())
            neg_frame = np.random.choice(neg_frames)
            pairs.append({
                'type': 'neg',
                'anchor_obj': obj,
                'anchor_frame': f1,
                'pair_obj': neg_obj,
                'pair_frame': neg_frame,
                'anchor_class': class_map[obj],
                'pair_class': class_map[neg_obj],
            })
    np.random.shuffle(pairs)
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate precomputed positive/negative pairs for Jeongok-ReID dataset.")
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata.pkl')
    parser.add_argument('--output', type=str, required=True, help='Output PKL file for pairs')
    parser.add_argument('--train', action='store_true', help='Use train split (default: val)')
    parser.add_argument('-K', type=int, default=200, help='Number of positive pairs per class')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print(f"Loading metadata from {args.metadata} (train={args.train})...")
    metadata, valid_targets = load_jeongok_metadata(args.metadata, train=args.train)
    print(f"Found {len(valid_targets)} valid targets.")
    print(f"Generating pairs (K={args.K})...")
    pairs = generate_pairs(metadata, valid_targets, K=args.K, seed=args.seed)
    print(f"Total pairs generated: {len(pairs)}")
    print(f"Saving pairs to {args.output} ...")
    with open(args.output, 'wb') as f:
        pickle.dump(pairs, f)
    print("Done.") 

# python3 tools/make_pairs_pkl.py --metadata Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/metadata/metadata.pkl --output Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/train_pairs.pkl --train
# python3 tools/make_pairs_pkl.py --metadata Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/metadata/metadata.pkl --output Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/val_pairs.pkl

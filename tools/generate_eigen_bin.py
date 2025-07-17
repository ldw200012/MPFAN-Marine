#!/usr/bin/env python3
"""
Generate pts_xyz_eigen_{k}.bin files for each frame in the Jeongok-ReID dataset.
For each point in pts_xyz.bin, compute the eigenvalues of the local covariance matrix
using its k-nearest neighbors, sort them in descending order, and save as pts_xyz_eigen_{k}.bin.
"""
import os
import numpy as np
from pathlib import Path
import argparse
from sklearn.neighbors import NearestNeighbors

def process_frame(pts_path, k):
    # Load points
    points = np.fromfile(pts_path, dtype=np.float32)
    if points.size % 3 != 0:
        print(f"[WARN] File {pts_path} size not divisible by 3, skipping.")
        return
    points = points.reshape(-1, 3)
    N = points.shape[0]
    if N < k:
        print(f"[WARN] Not enough points ({N}) for k={k} in {pts_path}, skipping.")
        return
    # kNN (include self)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    # For each point, compute eigenvalues of local covariance
    eigenvalues = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        neighbors = points[indices[i]]
        cov = np.cov(neighbors, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)  # ascending order
        eigvals = eigvals[::-1]  # descending order
        eigenvalues[i] = eigvals
    return eigenvalues

def main():
    parser = argparse.ArgumentParser(description="Generate pts_xyz_eigen_{k}.bin for each frame.")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root (should contain objects/)')
    parser.add_argument('-k', type=int, required=True, help='Number of neighbors for kNN')
    parser.add_argument('--overwrite', action='store_true', default=True,
                        help='Overwrite existing eigenvalue files (default: True)')
    args = parser.parse_args()

    objects_dir = Path(args.data_root) / 'objects'
    if not objects_dir.exists():
        print(f"[ERROR] Objects directory not found: {objects_dir}")
        return 1

    print(f"Scanning objects in {objects_dir}")
    for obj_id in sorted(os.listdir(objects_dir)):
        obj_path = objects_dir / obj_id
        if not obj_path.is_dir():
            continue
        for frame_id in sorted(os.listdir(obj_path)):
            frame_path = obj_path / frame_id
            if not frame_path.is_dir():
                continue
            pts_path = frame_path / 'pts_xyz.bin'
            if not pts_path.exists():
                print(f"[WARN] Missing {pts_path}, skipping.")
                continue
            eigen_path = frame_path / f'pts_xyz_eigen_{args.k}.bin'
            if eigen_path.exists() and not args.overwrite:
                print(f"[INFO] {eigen_path} exists, skipping.")
                continue
            print(f"Processing {pts_path} (k={args.k}) ...")
            eigenvalues = process_frame(pts_path, args.k)
            if eigenvalues is not None:
                points = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 3)

                # for i in range(points.shape[0]):
                #     x, y, z = points[i]
                #     eigs = eigenvalues[i]
                #     print(f"{x:.6f} {y:.6f} {z:.6f} {eigs[0]:.6e} {eigs[1]:.6e} {eigs[2]:.6e}")
                
                out = np.concatenate([points, eigenvalues], axis=1)  # shape (N, 6)
                out.astype(np.float32).tofile(eigen_path)
                print(f"  Saved: {eigen_path}")
            else:
                print(f"  [WARN] Skipped writing for {pts_path} due to insufficient points (N < k={args.k}) or malformed data.")
    print("Done.")
    return 0

if __name__ == '__main__':
    exit(main()) 
import os
import time
import numpy as np
import open3d as o3d

def visualize_point_cloud(bin_path):
    """Reads a KITTI-style .bin point cloud file."""
    point_cloud = np.fromfile(bin_path, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 3))  # x, y, z, intensity

    """Visualizes a point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    path = "/media/dongwooklee1201/Data_Storage_A1/Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/objects/target1"
    subfolders = sorted([subfolder for subfolder in os.listdir(path)]) #  if f.endswith('.bin')

    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        bin_files = [f for f in os.listdir(subfolder_path) if f.endswith('pts_xyz.bin')]

        if not bin_files:
            print("No .bin files found in the folder.")
        else:
            bin_file = bin_files[0]
            bin_path = os.path.join(subfolder_path, bin_file)
            print(f"Found {len(bin_files)} .bin files. Visualizing...")
            
            visualize_point_cloud(bin_path)
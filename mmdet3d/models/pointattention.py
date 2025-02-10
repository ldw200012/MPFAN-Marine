import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

class AttentionLayer(nn.Module):
    def __init__(self, point_features_dim, heads):
        super(AttentionLayer, self).__init__()

        self.point_features_dim = point_features_dim
        self.heads = heads
        self.head_dim = point_features_dim // heads

        assert (
            self.head_dim * heads == point_features_dim
        ), "Point features dimension must be divisible by the number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, point_features_dim)

    def forward(self, xyz):
        N, points, dim = xyz.size()

        values = self.values(xyz).view(N, points, self.heads, self.head_dim)
        keys = self.keys(xyz).view(N, points, self.heads, self.head_dim)
        queries = self.queries(xyz).view(N, points, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.point_features_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshapes(
            N, points, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        # Attention attention: torch.Size([64, 1, 128, 128])
        # attention shape: (batch_size, heads, num_points, num_points)
        return out, attention
    
class AdaptiveVoxelSize(nn.Module):
    def __init__(self, voxel_size_base, voxel_size_range):
        super(AdaptiveVoxelSize, self).__init__()

        self.voxel_size_base = voxel_size_base
        self.voxel_size_range = voxel_size_range

    def forward(self, xyz, attention_weights):
        N, points, dim = xyz.size()

        voxel_sizes = torch.zeros_like(xyz[:, :, 0])  # Assume all voxels initially have base size

        for i in range(N):
            attention_row = attention_weights[i]  # For each point in the point cloud
            # print(f"Attention Row [{i}] Shape ", attention_row.shape)
            # print(f"Attention Row [{i}]: ", attention_row)

            max_attention = torch.max(attention_row, dim=1)[0]  # Get the maximum attention for each point
            # print(f"Attention Row [{i}] MAX before: ", max_attention)
            # print(f"Attention Row [{i}] Range: ", (torch.max(max_attention) - torch.min(max_attention)))


            max_attention = (max_attention - torch.min(max_attention)) / (torch.max(max_attention) - torch.min(max_attention))
            # max_attention = (max_attention - torch.min(max_attention))
            # print(f"Attention Row [{i}] MAX after: ", max_attention)

            voxel_sizes[i] = self.voxel_size_base + (1 - max_attention) * self.voxel_size_range

        return voxel_sizes

class PointCloudAttention(nn.Module):
    def __init__(self, point_features_dim, heads, voxel_size_base, voxel_size_range):
        super(PointCloudAttention, self).__init__()

        self.attention_layer = AttentionLayer(point_features_dim, heads)
        self.adaptive_voxel_size = AdaptiveVoxelSize(voxel_size_base, voxel_size_range)
        
    def forward(self, batched_pts, mode='test'):
        xyz = batched_pts.transpose(2, 1)
        N, points, dim = xyz.size()

        for n in range(N):
            unique_rows_np = np.unique(xyz[n].cpu().numpy(), axis=0)
            unique_rows_tensor = torch.tensor(unique_rows_np)

            # print("\033[91m{}\033[0m: {}".format(f"XYZ Shape", unique_rows_tensor.shape))

            if (int(unique_rows_tensor.shape[0]) > 100):
                print("\033[91m{}\033[0m: {}".format(f"XYZ", unique_rows_tensor.shape))

                # numpy_array = unique_rows_tensor.cpu().numpy()
                # np.set_printoptions(threshold=np.inf)
                # print("\033[91m{}\033[0m: {}".format(f"XYZ", numpy_array))

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz[0].cpu())[:, :3])
        # o3d.visualization.draw_geometries([pcd])

        out, attention = self.attention_layer(xyz)

        voxel_sizes = self.adaptive_voxel_size(xyz, attention)
        # print("\033[91m{}\033[0m: {}".format(f"Voxel_sizes Shape", voxel_sizes.shape))
        # Voxel_sizes Shape: torch.Size([64, 128])
        # print("Voxel Sizes: ", voxel_sizes)

        min_value = torch.min(voxel_sizes)
        max_value = torch.max(voxel_sizes)
        # print("\033[91m{}\033[0m: {}".format(f"MAX, MIN", f"({max_value}, {min_value})"))
        # print("\033[91m{}\033[0m: {}".format(f"BINS", bins))

        # Create 10 bins within that range
        bin_size = 10
        bins = torch.linspace(float(min_value), float(max_value), bin_size+1)

        bins = bins.to(voxel_sizes.device)  # Move bins to the same device as voxel_sizes
        splitted_point_clouds = [torch.tensor([], dtype=torch.float) for _ in range(bin_size)]

        for i in range(N):  # Loop through each point cloud in the batch
            for j in range(points):  # Loop through each point in the point cloud
                bin_index = torch.bucketize(voxel_sizes[i, j], bins, right=True) - 1

                if 0 <= bin_index < len(splitted_point_clouds):
                    splitted_point_clouds[bin_index] = torch.cat((splitted_point_clouds[bin_index].cuda(), xyz[i, j].unsqueeze(0)))
                else:
                    splitted_point_clouds[bin_size-1] = torch.cat((splitted_point_clouds[bin_size-1].cuda(), xyz[i, j].unsqueeze(0)))

        # for idx, spc in enumerate(splitted_point_clouds):
            # print("\033[91m{}\033[0m: {}".format(f"Split Points [{idx}] Shape", spc.shape))


        return None


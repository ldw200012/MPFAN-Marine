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

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, points, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        # Attention attention: torch.Size([64, 1, 128, 128])
        # attention shape: (batch_size, heads, num_points, num_points)
        return out, attention
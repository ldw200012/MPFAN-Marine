"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import numpy as np
import torch
from torch.nn import Module, Dropout

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  
        KV = torch.einsum("nshd,nshv->nhdv", K, values) 
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

"""
code Taken from  ST Net repository  https://github.com/fpthink/STNet/blob/main/modules/pointnet2_utils.py
"""
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(Self_Attention, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, feat, xyz, mask=None):
        """
        Args:
            feat (torch.Tensor):    [B, C, N]
            xyz (torch.Tensor):     [B, N, 3]
            mask (torch.Tensor):    [B, N] (optional)
        """
        bs = feat.size(0)
        feat = feat.permute(0, 2, 1)

        mlp_xyz = self.pos_mlp(xyz)
        feat_pos = feat + mlp_xyz
        
        # multi-head attention
        query = self.q_proj(feat_pos).view(bs, -1, self.nhead, self.dim)                # [B, N, (H, D)]
        key = self.k_proj(feat_pos).view(bs, -1, self.nhead, self.dim)                  # [B, N, (H, D)]
        value = self.v_proj(feat_pos).view(bs, -1, self.nhead, self.dim)                # [B, N, (H, D)]

        message = self.attention(query, key, value, q_mask=mask, kv_mask=mask)      # [B, N, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))             # [B, N, C=H*D]
        message = self.norm1(message)                                               # [B, N, C=H*D]

        # feed-forward network
        message = self.mlp(torch.cat([feat, message], dim=2))                       # [B, N, C=H*D]
        message = self.norm2(message)                                               # [B, N, C=H*D]

        return (feat + message).permute(0, 2, 1)    # [B, C, N]
    
    def forward_subsample(self, feat, xyz, mask=None):
        bs = feat.size(0)
        feat = feat.permute(0, 2, 1)

        # print("bs, feat, xyz: ", bs, feat.shape, xyz.shape)
        mlp_xyz = self.pos_mlp(xyz)
        # print("mlp_xyz: ", mlp_xyz.shape)

        feat_pos = feat + mlp_xyz

        # multi-head attention
        query = self.q_proj(feat_pos).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(feat_pos).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(feat_pos).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=mask, kv_mask=mask)
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([feat, message], dim=2))
        message = self.norm2(message)

        # After attention and feed-forward, compute weights for each point
        output_feat = (feat + message).permute(0, 2, 1)  # Bringing it back to [B, C, N]
        point_weights = torch.norm(output_feat, p=2, dim=1)  # Compute L2 norm across channel dimension for weights

        # Now, output_feat contains the transformed features, and point_weights contains the weights of each point.
        # You can return point_weights if you need it for further processing.
        return point_weights

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def fps_subset_split(xyz, npoint):
    """
    xyz: [B, N, 3] array containing the whole point cloud
    npoint: samples you want in the sampled point cloud typically << N 
    """
    device = xyz.device
    B, N, C = xyz.shape
    subset1 = torch.zeros((B, int(N/2), C)).to(device)
    subset2 = torch.zeros((B, int(N/2), C)).to(device)

    for b in range(B):
        points = xyz[b].cpu()

        points_left = np.arange(len(points)) # [P]
        sample_inds = np.zeros(npoint, dtype='int') # [S]
        dists = np.ones_like(points_left) * float('inf') # [P]

        selected = np.random.randint(N)
        sample_inds[0] = points_left[selected]

        points_left = np.delete(points_left, selected) # [P - 1]
        for i in range(1, npoint):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]
            
            dist_to_last_added_point = ((points[last_added] - points[points_left])**2).sum(-1) # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left]) # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        mask = np.zeros(points.shape[0], dtype=bool)
        mask[sample_inds] = True

        # print("Batch ", b, ": ", sample_inds)
        subset1[b] = points[mask]
        subset2[b] = points[~mask]

    return subset1, subset2

def random_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: random sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    centroids = torch.arange(npoint, dtype=torch.long).to(device).repeat(xyz.size(0), 1)

    # print("centroid shape: ", centroids.shape)
    # print("centroids: ", centroids)
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """

    B, N, _ = src.shape
    _, M, _ = dst.shape


    # print("\033[91mHERE\033[0m")

    # dist = torch.zeros(B,N,M)
    # print("\033[91mAFTER\033[0m")
    # # print(dist)
    # # print(dist.device)

    # # src.cpu()
    # # print(src.device)
    # # # print(torch.sum(src ** 2, -1).view(B, N, 1))
    # # # print(torch.sum(dst ** 2, -1).view(B, 1, M))
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # print(f"4-7) square_distance : src: {src.shape} | dst: {dst.permute(0, 2, 1).shape}") # delete-print
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    return dist

def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    if dim < 0:
        dim += input.ndim
    
    transpose_dims = [i for i in range(input.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    input = input.permute(transpose_dims)
    index = torch.argsort(input, dim=0, descending=largest)
    indices = index[:k]
    indices = indices.permute(transpose_dims)
    return [None, indices]

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    # print("\033[91mFirst Square Distance\033[0m")
    # print("\033[91mnew_xyz\033[0m", new_xyz)
    # print("\033[91mxyz\033[0m", xyz)
    # print("4-6) knn_point") # delete-print
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)

    # print("\033[91mSecond Square Distance\033[0m") # delete-print

    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group_edge(npoint, radius, nsample, xyz, points, sampling, numpoints, returnfps=False, use_knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = numpoints

    if sampling == "FPS":
        fps_idx = farthest_point_sample(xyz, numpoints) # [B, numpoints]
    elif sampling == "RANDOM":
        fps_idx = random_point_sample(xyz, numpoints) # [B, numpoints]
    elif sampling == "FULL":
        device = xyz.device
        fps_idx = torch.arange(N, dtype=torch.long).to(device).repeat(xyz.size(0), 1)

    new_xyz = index_points(xyz, fps_idx)                    # B x S x 3
    # print('4-5) SA in:',xyz.shape, points.shape if points is not None else None, numpoints) # delete-print
    # print("4-5a) Check var validity 'xyz': ", xyz) # delete-print
    # print("4-5b) Check var validity 'points': ", points) # delete-print
    # print("4-5c) Check var validity 'numpoints': ", numpoints) # delete-print
    
    if use_knn:
        idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)   # B x S x K

    if sampling == "FULL":
        grouped_xyz = index_points(xyz, idx)                        # B x N x k x 3
        # print("group: ", grouped_xyz.shape)
        # print("new_xyz: ", new_xyz.shape)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, N, 1, C)   # B x N x k x 3
    else:
        grouped_xyz = index_points(xyz, idx)                        # B x S x k x 3
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)   # B x S x k x 3

    # print("grouped_xyz_norm: ", grouped_xyz_norm.shape)


    if points is not None:
        # points:           B x N x D
        # fps_idx:          B x S
        # center_points:    B x S x D
        center_points = index_points(points, fps_idx)           # B x S x D
        grouped_points = index_points(points, idx)

        new_points = torch.cat([grouped_xyz_norm, 
                                center_points.unsqueeze(2).repeat([1, 1, nsample, 1]),
                                grouped_points-center_points.unsqueeze(2)],
                            dim=-1)
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstractionEdgeSA(nn.Module):
    def __init__(self, npoint, radius, nsample, mlp, sampling, use_xyz=True, group_all=False, use_knn=False):
        super(PointNetSetAbstractionEdgeSA, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.sampling = sampling
        self.use_knn = use_knn

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        if self.use_xyz:
            mlp[0] += 3

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.self_attention = Self_Attention(last_channel, 2, 'linear')

    def forward(self, xyz, points, numpoints):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points_concat: sample points feature data, [B, D', S]
        """
        ##### print("4-3) point input to SA_modules (xyz, points, numpoints): ", xyz.shape, points.shape if points is not None else None, numpoints) # delete-print
        ##### print("4-3a) Check var validity 'xyz': ", xyz) # delete-print
        ##### print("4-3b) Check var validity 'points': ", points) # delete-print
        ##### print("4-3c) Check var validity 'numpoints': ", numpoints) # delete-print
        # xyz = xyz.permute(0, 2, 1)    # le
        if points is not None:
            points = points.permute(0, 2, 1)    # BxNxD

        if self.group_all:
            # print("4-4) OPTION 1 [sample_and_group_all]") # delete-print
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # print("4-4) OPTION 2 [sample_and_group_edge]") # delete-print
            new_xyz, new_points = sample_and_group_edge(self.npoint, self.radius, self.nsample, xyz, points, self.sampling, numpoints, False, self.use_knn)

        new_points = new_points.permute(0, 3, 1, 2)     # [B, D, numpoints, nsample]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 3)[0]        # B x D x numpoints

        new_points = self.self_attention(new_points, new_xyz)
        return new_xyz, new_points

class FP_SA(nn.Module):
    def __init__(self,
                 last_channel,
                 feat1_dim,     # B x C1 x N
                 feat2_dim,     # B x C2 x S
                 d_model,
                 out_dim,
                 nhead,
                 attention='linear'):
        super(FP_SA, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # position encoding
        # self.pos_mlp1 = nn.Sequential(
        #     nn.Linear(3, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, feat1_dim)
        # )

        self.pos_mlp2 = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feat2_dim)
        )

        # multi-head attention
        self.q_proj = nn.Linear(feat1_dim, d_model, bias=False)         # feat1
        self.k_proj = nn.Linear(feat2_dim, d_model, bias=False)         # feat2
        self.v_proj = nn.Linear(feat2_dim, d_model, bias=False)         # feat2
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(feat1_dim+d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, out_dim, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, feat1, xyz1, feat2, xyz2, mask=None):
        """
        Args:
            feat1 (torch.Tensor):    [B, C1, N]
            xyz1 (torch.Tensor):     [B, N, 3]
            feat2 (torch.Tensor):    [B, C2, S]
            xyz2 (torch.Tensor):     [B, S, 3]
            mask (torch.Tensor):    [B, N] (optional)
        """
        bs = feat1.size(0)

        feat1 = feat1.permute(0, 2, 1)  # [B, N, C1]
        feat2 = feat2.permute(0, 2, 1)  # [B, S, C2]

        # feat1 = feat1 + self.pos_mlp1(xyz1)
        feat2_pos = feat2 + self.pos_mlp2(xyz2)

        # multi-head attention
        query = self.q_proj(feat1).view(bs, -1, self.nhead, self.dim)            # [B, N, (H, D)]
        key = self.k_proj(feat2).view(bs, -1, self.nhead, self.dim)              # [B, N, (H, D)]
        value = self.v_proj(feat2_pos).view(bs, -1, self.nhead, self.dim)            # [B, N, (H, D)]

        message = self.attention(query, key, value, q_mask=None, kv_mask=None)      # [B, N, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))             # [B, N, C=H*D]
        message = self.norm1(message)                                               # [B, N, C=H*D]

        # feed-forward network
        message = self.mlp(torch.cat([feat1, message], dim=2))                      # [B, N, C=H*D]
        message = self.norm2(message)

        ##### print("4-4) message.permute(0, 2, 1) shape: ", message.permute(0, 2, 1).shape)
        ##### print("----------------------------------------------------------------------")
        return message.permute(0, 2, 1)         # [B, C, N]

class PointNetFeaturePropagationSA(nn.Module):
    def __init__(self, mlp, mlp_inte):
        super(PointNetFeaturePropagationSA, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.interpolation = FP_SA(
                                    last_channel=mlp_inte[0],
                                    feat1_dim=mlp_inte[1],
                                    feat2_dim=mlp_inte[2],
                                    d_model=mlp_inte[3],
                                    out_dim=mlp_inte[4],
                                    nhead=2,
                                    attention='linear')

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # print("xyz1 shape: ", xyz1.shape) ### delete-print
        # print("xyz2 shape: ", xyz2.shape) ### delete-print
        # print("points1: ", points1.shape) ### delete-print
        # print("points2: ", points2.shape) ### delete-print
        B, N, C = xyz1.shape
        
        inte_points1 = self.interpolation(points1, xyz1, points2, xyz2)     # [B, C2, N]
        return inte_points1
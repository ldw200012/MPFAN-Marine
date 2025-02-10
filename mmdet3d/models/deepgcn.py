import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import knn_point
from openpoints.models.backbone import DeepGCNEncoder

class DeepGCN(nn.Module):
    def __init__(self, emb_dims=1024):
        super(DeepGCN, self).__init__()
        print("\033[91mDeepGCN Created\033[0m")

        torch.cuda.synchronize()

        in_channels = 3
        self.encoder = DeepGCNEncoder(in_channels=in_channels, channels=64, emb_dims=emb_dims, n_blocks=14, # n_blocks=14
                                      conv='edge', block='no', k=16, epsilon=0.2, #block='res'
                                      use_stochastic=True, use_dilation=True,
                                      norm_args={'norm': 'bn'}, act_args={'act': 'relu'}, conv_args={'order': 'conv-norm-act'},
                                      is_seg=False)
                                      
    def forward(self, data, numpoints):
        _, f = self.encoder.forward_seg_feat(data)
        return data, f

class ED_DeepGCN(nn.Module):
    def __init__(self, emb_dims=1024, ED_nsample=10, ED_conv_out=4):
        super(ED_DeepGCN, self).__init__()
        print("\033[91mED_DeepGCN Created\033[0m")
        
        torch.cuda.synchronize()

        in_channels = 3
        self.encoder = DeepGCNEncoder(in_channels=in_channels, channels=64, emb_dims=emb_dims, n_blocks=14, # n_blocks=14
                                      conv='edge', block='no', k=16, epsilon=0.2, #block='res'
                                      use_stochastic=True, use_dilation=True,
                                      norm_args={'norm': 'bn'}, act_args={'act': 'relu'}, conv_args={'order': 'conv-norm-act'},
                                      is_seg=False)
        
        # Eigen ###############################################################################################################
        self.ED_nsample = ED_nsample
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out))
        
        # Final ###############################################################################################################
        self.conv_final = nn.Conv1d(emb_dims + ED_conv_out, emb_dims, 1)
        self.bn_final = nn.BatchNorm1d(emb_dims)
                                      
    def forward(self, data, numpoints):
        _, f = self.encoder.forward_seg_feat(data)

        # Eigen ###############################################################################################################
        group_idx = knn_point(nsample=self.ED_nsample, xyz=data, new_xyz=data)
        batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1).expand(-1, data.shape[1], self.ED_nsample)
        neighborhood_points = data[batch_indices, group_idx]  # (B, N, k, 3)
        centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
        cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)
        eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
        eigen_feature = self.sub3_ED(eigenvalues)

        # Final ###############################################################################################################
        z = torch.cat((f, eigen_feature.permute(0,2,1)), dim=1)
        z = F.relu(self.bn_final(self.conv_final(z)))
        
        return data, z

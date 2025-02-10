"""
code below this line is taken/modified from https://github.com/fpthink/STNet/blob/main/modules/pointnet2_utils.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstractionEdgeSA, PointNetFeaturePropagationSA, knn_point

class Pointnet_Backbone(nn.Module):

    def __init__(self, input_channels=3, use_xyz=True, conv_out=32, mul=1, radius=[0.3,0.5,0.7], nsample=[32,48,48]):
        super(Pointnet_Backbone, self).__init__()
        print("\033[91mPointTransformer Created\033[0m")

        k = ()
        mul = mul
        sa1 = 32 * mul
        sa2 = 64 * mul
        sa3 = 128 * mul

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[0],
                nsample=nsample[0],
                mlp=[input_channels, sa1, sa1, sa1],
                sampling="FPS",
                use_xyz=use_xyz,
                use_knn=True
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[1],
                nsample=nsample[1],
                mlp=[sa2, sa2, sa2, sa2],
                sampling="FPS",
                use_xyz=use_xyz,
                use_knn=True
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[2],
                nsample=nsample[2],
                mlp=[sa3, sa3, sa3, sa3],
                sampling="FPS",
                use_xyz=use_xyz,
                use_knn=True
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[67,sa1,sa1], mlp_inte=[sa2, 3, sa2, sa2, sa1]))
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[160,sa3,sa2], mlp_inte=[sa3, sa1, sa3, sa2, sa2]))    # 160=128+32
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[192,sa3,sa3], mlp_inte=[sa3, sa2, sa3, sa2, sa3]))  # 192=128+64

        self.cov_final = nn.Conv1d(sa1, conv_out, kernel_size=1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints):

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        l_features[0] = xyz.transpose(1, 2).contiguous()
        for i in [2, 1, 0]:
            l_features[i] = self.FP_modules[i](l_xyz[i], l_xyz[i+1], l_features[i], l_features[i+1])

        out1 = l_xyz[0]
        out2 = self.cov_final(l_features[0])

        return out1, out2  # [B, N, 3], [B, conv_out=32, N]
    
class ED_Pointnet_Backbone(nn.Module):

    def __init__(self, input_channels=3, use_xyz=True, conv_out=32, mul=1, radius=[0.3,0.5,0.7], nsample=[32,48,48], ED_nsample=10, ED_conv_out=8):
        super(ED_Pointnet_Backbone, self).__init__()
        print("\033[91mEDPointTransformer Created\033[0m")

        k = ()
        mul = mul
        sa1 = 32 * mul
        sa2 = 64 * mul
        sa3 = 128 * mul

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[0],
                nsample=nsample[0],
                mlp=[input_channels, sa1, sa1, sa1],
                sampling="FPS",
                use_xyz=use_xyz,
                use_knn=True,
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[1],
                nsample=nsample[1],
                mlp=[sa2, sa2, sa2, sa2],
                sampling="FPS",
                use_xyz=use_xyz,
                use_knn=True,
            )
        )
        self.SA_modules.append(
            PointNetSetAbstractionEdgeSA(
                npoint=None,
                radius=radius[2],
                nsample=nsample[2],
                mlp=[sa3, sa3, sa3, sa3],
                sampling="FPS",
                use_xyz=use_xyz,
                use_knn=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[67,sa1,sa1], mlp_inte=[sa2, 3, sa2, sa2, sa1]))
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[160,sa3,sa2], mlp_inte=[sa3, sa1, sa3, sa2, sa2]))    # 160=128+32
        self.FP_modules.append(PointNetFeaturePropagationSA(mlp=[192,sa3,sa3], mlp_inte=[sa3, sa2, sa3, sa2, sa3]))  # 192=128+64

        self.cov_final = nn.Conv1d(sa1, conv_out, kernel_size=1)

        # Eigen ###############################################################################################################
        self.ED_nsample = ED_nsample
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out))
        
        # Final ###############################################################################################################
        self.conv1 = nn.Conv1d(conv_out + ED_conv_out, conv_out, 1)
        self.bn1 = nn.BatchNorm1d(conv_out)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints):

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        l_features[0] = xyz.transpose(1, 2).contiguous()
        for i in [2, 1, 0]:
            l_features[i] = self.FP_modules[i](l_xyz[i], l_xyz[i+1], l_features[i], l_features[i+1])

        out1 = l_xyz[0]
        out2 = self.cov_final(l_features[0])

        # Eigen ###############################################################################################################
        group_idx = knn_point(nsample=self.ED_nsample, xyz=xyz, new_xyz=xyz)
        batch_indices = torch.arange(xyz.shape[0]).view(-1, 1, 1).expand(-1, xyz.shape[1], self.ED_nsample)
        neighborhood_points = xyz[batch_indices, group_idx]  # (B, N, k, 3)
        centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
        cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)
        eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
        eigen_feature = self.sub3_ED(eigenvalues)

        # Final ###############################################################################################################
        z = torch.cat((out2, eigen_feature.permute(0,2,1)), dim=1)
        z = F.relu(self.bn1(self.conv1(z)))

        return out1, z  # [B, N, 3], [B, conv_out=32, N]
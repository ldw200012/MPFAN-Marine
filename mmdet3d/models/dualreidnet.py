import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import knn_point

from .pointnet import PointNet, ED_PointNet
from .pointnext import PointNeXt, ED_PointNeXt
from .dgcnn_orig import DGCNN, ED_DGCNN
from .deepgcn import DeepGCN, ED_DeepGCN
from .backbone_net import Pointnet_Backbone, ED_Pointnet_Backbone
from .spotr import SPoTr, ED_SPoTr

# class DualReID(nn.Module):
#     def __init__(self, fe_module='pointnet'):
#         super(DualReID, self).__init__()
#         # torch.cuda.synchronize()

#         self.FE = None
#         self.fe_module = fe_module
#         graph_emb_dims = 1024
#         if fe_module == 'pointnet':
#             self.FE = PointNet(k=40,normal_channel=False)
#         elif fe_module == 'pointnext':
#             self.FE = PointNeXt()
#         elif fe_module == 'dgcnn':
#             self.FE = DGCNN(dropout=0.5,emb_dims=graph_emb_dims, k=20, output_channels=40)
#         elif fe_module == 'deepgcn':
#             self.FE = DeepGCN(emb_dims=graph_emb_dims)
#         else:
#             raise Exception("fe_module must be one of the followings: ['pointnet', 'pointnext', 'dgcnn', 'deepgcn']")
        
#         self.SA = Pointnet_Backbone(input_channels=0, use_xyz=True, conv_out=128, nsample=[16,16,16])

#         # self.FE = ED_DGCNN(dropout=0.5,emb_dims=graph_emb_dims, k=20, output_channels=40, ED_nsample=10, ED_conv_out=4)
#         # self.SA = ED_Pointnet_Backbone(input_channels=0, use_xyz=True, conv_out=128, nsample=[16,16,16], ED_nsample=10, ED_conv_out=8)

#         # FE --> 32
#         self.FE_conv1 = nn.Conv1d(graph_emb_dims, 256, 1)
#         self.FE_conv2 = nn.Conv1d(256, 64, 1)
#         self.FE_conv3 = nn.Conv1d(64, 32, 1)
#         self.FE_bn1 = nn.BatchNorm1d(256)
#         self.FE_bn2 = nn.BatchNorm1d(64)
#         self.FE_bn3 = nn.BatchNorm1d(32)

#         # Concat --> 128
#         self.conv1 = nn.Conv1d(128 + 32, 512, 1)
#         self.conv2 = nn.Conv1d(512, 256, 1)
#         self.conv3 = nn.Conv1d(256, 128, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)

#     def _break_up_pc(self, pc):
#         xyz = pc[..., 0:3].contiguous()
#         features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
#         return xyz, features
    
#     def forward(self, pointcloud, numpoints):
#         xyz, _ = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

#         _, h1 = self.SA(xyz, numpoints)                      # (B, conv_out, N)

#         h2 = None
#         if self.fe_module in ['dgcnn', 'pointnet']:
#             _, h2 = self.FE(xyz.permute(0,2,1), numpoints)
#         else: # pointnext, deepgcn
#             _, h2 = self.FE(xyz, numpoints)

#         h2_ = None
#         if self.fe_module == 'pointnext':
#             h2_ =  F.relu(self.FE_bn3(self.FE_conv3(h2)))
#         else:
#             h2_ =  F.relu(self.FE_bn1(self.FE_conv1(h2)))
#             h2_ =  F.relu(self.FE_bn2(self.FE_conv2(h2_)))
#             h2_ =  F.relu(self.FE_bn3(self.FE_conv3(h2_)))

#         # print("DualReID ({}) | h1:{}, h2:{}".format(self.fe_module, h1.shape, h2_.shape))
#         z = torch.cat((h1, h2_), dim=1)

#         z_ = F.relu(self.bn1(self.conv1(z)))
#         z_ = F.relu(self.bn2(self.conv2(z_)))
#         z_ = F.relu(self.bn3(self.conv3(z_)))
    
#         return xyz, z_ # [B, N/2, 3], [B, 128, N/2]

######################################################
# DualReID(DGCNN) ==> DGCloneXt
######################################################

class DualReID(nn.Module):
    def __init__(self, input_channels=0, use_xyz=True, SA_conv_out=128, conv_out=128, nsample=[16,16,16], fe_module='pointnet'):
        super(DualReID, self).__init__()
        # torch.cuda.synchronize()

        self.sub1_SA = Pointnet_Backbone(input_channels=0, use_xyz=True, conv_out=SA_conv_out, nsample=nsample)
        self.sub2_DG = DGCNN(dropout=0.5,emb_dims=1024, k=20, output_channels=40) # output = emb_dims = 1024

         # 1024 to 32 for DGCNN
        self.DG_conv1 = nn.Conv1d(1024, 256, 1)
        self.DG_conv2 = nn.Conv1d(256, 64, 1)
        self.DG_conv3 = nn.Conv1d(64, int(SA_conv_out/4), 1)

        self.DG_bn1 = nn.BatchNorm1d(256)
        self.DG_bn2 = nn.BatchNorm1d(64)
        self.DG_bn3 = nn.BatchNorm1d(int(SA_conv_out/4))

        self.conv1 = nn.Conv1d(SA_conv_out + int(SA_conv_out/4), 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, conv_out, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(conv_out)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints):
        xyz, features = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

        # Clone 1 through Self-Attention
        out1, h1 = self.sub1_SA(xyz, numpoints)     # h1 shape = [B, conv_out, N]
        
        # Clone 2 through DGCNN
        out2, h2 = self.sub2_DG(xyz.permute(0,2,1), numpoints)
        h2_ =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2_ =  F.relu(self.DG_bn2(self.DG_conv2(h2_)))
        h2_ =  F.relu(self.DG_bn3(self.DG_conv3(h2_)))

        # out2 = out2.permute(0,2,1)
        z = torch.cat((h1, h2_), dim=1)

        z_ = F.relu(self.bn1(self.conv1(z)))
        z_ = F.relu(self.bn2(self.conv2(z_)))
        z_ = F.relu(self.bn3(self.conv3(z_)))
        
        return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]
    

# class ED_DualReID(nn.Module):
#     def __init__(self, fe_module='pointnet', ED_nsample=10, ED_conv_out=4):
#         super(ED_DualReID, self).__init__()
#         torch.cuda.synchronize()

#         self.FE = None
#         self.fe_module = fe_module
#         graph_emb_dims = 1024
#         if fe_module == 'pointnet':
#             self.FE = PointNet(k=40,normal_channel=False)
#         elif fe_module == 'pointnext':
#             self.FE = PointNeXt()
#         elif fe_module == 'dgcnn':
#             self.FE = DGCNN(dropout=0.5,emb_dims=graph_emb_dims, k=20, output_channels=40)
#         elif fe_module == 'deepgcn':
#             self.FE = DeepGCN(emb_dims=graph_emb_dims)
#         else:
#             raise Exception("fe_module must be one of the followings: ['pointnet', 'pointnext', 'dgcnn', 'deepgcn']")
        
#         self.SA = Pointnet_Backbone(input_channels=0, use_xyz=True, conv_out=128, nsample=[16,16,16])

#         # FE --> 32
#         self.FE_conv1 = nn.Conv1d(graph_emb_dims, 256, 1)
#         self.FE_conv2 = nn.Conv1d(256, 64, 1)
#         self.FE_conv3 = nn.Conv1d(64, 32, 1)
#         self.FE_bn1 = nn.BatchNorm1d(256)
#         self.FE_bn2 = nn.BatchNorm1d(64)
#         self.FE_bn3 = nn.BatchNorm1d(32)

#         # Concat --> 128
#         self.conv1 = nn.Conv1d(128 + 32, 512, 1)
#         self.conv2 = nn.Conv1d(512, 256, 1)
#         self.conv3 = nn.Conv1d(256, 128, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)

#         # Eigen ###############################################################################################################
#         self.ED_nsample = ED_nsample
#         self.ED_conv_out = ED_conv_out
#         self.sub3_ED = nn.Sequential(
#                             nn.Linear(3, ED_conv_out),
#                             nn.ReLU(),
#                             nn.Linear(ED_conv_out, ED_conv_out))
#         # self.sub3_ED = nn.Sequential(
#         #                     nn.Conv1d(3, ED_conv_out, 1),
#         #                     # nn.BatchNorm1d(ED_conv_out),
#         #                     nn.ReLU(),
#         #                     nn.Conv1d(ED_conv_out, ED_conv_out, 1),
#         #                     # nn.BatchNorm1d(ED_conv_out)
#         #                 )
        
#         # Final ###############################################################################################################
#         self.conv_final = nn.Conv1d(128 + ED_conv_out, 128, 1)
#         self.bn_final = nn.BatchNorm1d(128)

#     def _break_up_pc(self, pc):
#         xyz = pc[..., 0:3].contiguous()
#         features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
#         return xyz, features
    
#     def forward(self, pointcloud, numpoints):
#         xyz, _ = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

#         _, h1 = self.SA(xyz, numpoints)                      # (B, conv_out, N)

#         h2 = None
#         if self.fe_module in ['dgcnn', 'pointnet']:
#             _, h2 = self.FE(xyz.permute(0,2,1), numpoints)
#         else: # pointnext, deepgcn
#             _, h2 = self.FE(xyz, numpoints)

#         h2_ = None
#         if self.fe_module == 'pointnext':
#             h2_ =  F.relu(self.FE_bn3(self.FE_conv3(h2)))
#         else:
#             h2_ =  F.relu(self.FE_bn1(self.FE_conv1(h2)))
#             h2_ =  F.relu(self.FE_bn2(self.FE_conv2(h2_)))
#             h2_ =  F.relu(self.FE_bn3(self.FE_conv3(h2_)))

#         # print("DualReID ({}) | h1:{}, h2:{}".format(self.fe_module, h1.shape, h2_.shape))
#         f = torch.cat((h1, h2_), dim=1)

#         f_ = F.relu(self.bn1(self.conv1(f)))
#         f_ = F.relu(self.bn2(self.conv2(f_)))
#         f_ = F.relu(self.bn3(self.conv3(f_)))

#         # Eigen ###############################################################################################################
#         group_idx = knn_point(nsample=self.ED_nsample, xyz=xyz, new_xyz=xyz)
#         batch_indices = torch.arange(xyz.shape[0]).view(-1, 1, 1).expand(-1, xyz.shape[1], self.ED_nsample)
#         neighborhood_points = xyz[batch_indices, group_idx]  # (B, N, k, 3)
#         centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
#         cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)
#         eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
#         eigen_feature = self.sub3_ED(eigenvalues)

#         # print("eigen_feature: {}".format(eigen_feature.shape))

#         # Final ###############################################################################################################
#         z = torch.cat((f_, eigen_feature.permute(0,2,1)), dim=1)
#         z_ = F.relu(self.bn_final(self.conv_final(z)))
        
#         return xyz, z_ # [B, N/2, 3], [B, 128, N/2]
    

######################################################
# DualReID(DGCNN)-Eigen ==> DGCloneXt-Eigen
######################################################

class ED_DualReID(nn.Module):
    def __init__(self, input_channels=0, use_xyz=True, SA_conv_out=128, conv_out=128, nsample=[16,16,16], fe_module='pointnet', ED_nsample=10, ED_conv_out=4):
        super(ED_DualReID, self).__init__()
        # torch.cuda.synchronize()

        self.sub1_SA = Pointnet_Backbone(input_channels=0, use_xyz=True, conv_out=SA_conv_out, nsample=nsample)
        self.sub2_DG = DGCNN(dropout=0.5,emb_dims=1024, k=20, output_channels=40) # output = emb_dims = 1024

         # 1024 to 32 for DGCNN
        self.DG_conv1 = nn.Conv1d(1024, 256, 1)
        self.DG_conv2 = nn.Conv1d(256, 64, 1)
        self.DG_conv3 = nn.Conv1d(64, int(SA_conv_out/4), 1)

        self.DG_bn1 = nn.BatchNorm1d(256)
        self.DG_bn2 = nn.BatchNorm1d(64)
        self.DG_bn3 = nn.BatchNorm1d(int(SA_conv_out/4))

        self.conv1 = nn.Conv1d(SA_conv_out + int(SA_conv_out/4), 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, conv_out, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(conv_out)

        # Eigen ###############################################################################################################
        self.ED_nsample = ED_nsample
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out))
        
        # Final ###############################################################################################################
        self.conv_final = nn.Conv1d(128 + ED_conv_out, 128, 1)
        self.bn_final = nn.BatchNorm1d(128)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints):
        xyz, features = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

        # Clone 1 through Self-Attention
        out1, h1 = self.sub1_SA(xyz, numpoints)     # h1 shape = [B, conv_out, N]
        
        # Clone 2 through DGCNN
        out2, h2 = self.sub2_DG(xyz.permute(0,2,1), numpoints)
        h2_ =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2_ =  F.relu(self.DG_bn2(self.DG_conv2(h2_)))
        h2_ =  F.relu(self.DG_bn3(self.DG_conv3(h2_)))

        # out2 = out2.permute(0,2,1)
        f = torch.cat((h1, h2_), dim=1)

        f_ = F.relu(self.bn1(self.conv1(f)))
        f_ = F.relu(self.bn2(self.conv2(f_)))
        f_ = F.relu(self.bn3(self.conv3(f_)))

        # Eigen ###############################################################################################################
        group_idx = knn_point(nsample=self.ED_nsample, xyz=xyz, new_xyz=xyz)
        batch_indices = torch.arange(xyz.shape[0]).view(-1, 1, 1).expand(-1, xyz.shape[1], self.ED_nsample)
        neighborhood_points = xyz[batch_indices, group_idx]  # (B, N, k, 3)
        centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
        cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)
        eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
        eigen_feature = self.sub3_ED(eigenvalues)

        # print("eigen_feature: {}".format(eigen_feature.shape))

        # Final ###############################################################################################################
        z = torch.cat((f_, eigen_feature.permute(0,2,1)), dim=1)
        z_ = F.relu(self.bn_final(self.conv_final(z)))
        
        return xyz, z_, h1, h2_, eigen_feature.permute(0,2,1) # [B, N/2, 3], [B, 128, N/2]
        
        # return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]
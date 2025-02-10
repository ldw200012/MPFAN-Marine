import torch.nn as nn
from mmdet3d.models import FUSIONMODELS
from transformers import (AutoImageProcessor, 
                          AutoConfig, 
                          AutoModel, 
                          BeitModel, 
                          AutoFeatureExtractor, 
                          ViTForImageClassification, 
                          DeiTForImageClassificationWithTeacher)
import torch.nn.functional as F

from .lanegcn_nets import PostRes,LinearRes
from .pointattention import PointCloudAttention


from .pointnet import PointNet, ED_PointNet
from .pointnext import PointNeXt, ED_PointNeXt
from .dgcnn_orig import DGCNN, ED_DGCNN
from .deepgcn import DeepGCN, ED_DeepGCN
from .backbone_net import Pointnet_Backbone, ED_Pointnet_Backbone
from .spotr import SPoTr, ED_SPoTr
from .dualreidnet import DualReID, ED_DualReID

from mmdet.models import BaseDetector
import torch.distributed as dist
from pytorch3d.loss import chamfer_distance

import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

import torch
import copy
import time 

from .attention import corss_attention, local_self_attention, cross_lin_attn

module_obj = {
    'Linear':nn.Linear,
    'ReLU':nn.ReLU,
    'LSTM':nn.LSTM,
    'GroupNorm':nn.GroupNorm,
    'Embedding':nn.Embedding,
    'LayerNorm':nn.LayerNorm,
    'PostRes':PostRes,
    'LinearRes':LinearRes,
    'Pointnet_Backbone':Pointnet_Backbone,
    'corss_attention':corss_attention,
    'local_self_attention':local_self_attention,
    'Conv1d':nn.Conv1d,
    'Conv2d':nn.Conv2d,
    'BatchNorm1d':nn.BatchNorm1d,
    'Sigmoid':nn.Sigmoid,
    'cross_lin_attn':cross_lin_attn,
    'PointCloudAttention':PointCloudAttention,

    'PointNet':PointNet,
    'ED_PointNet':ED_PointNet,
    'PointNeXt':PointNeXt,
    'ED_PointNeXt':ED_PointNeXt,
    'DGCNN':DGCNN,
    'ED_DGCNN':ED_DGCNN,
    'DeepGCN':DeepGCN,
    'ED_DeepGCN':ED_DeepGCN,
    'ED_Pointnet_Backbone':ED_Pointnet_Backbone,
    'SPoTr':SPoTr,
    'ED_SPoTr':ED_SPoTr,
    'DualReID':DualReID,
    'ED_DualReID':ED_DualReID,
}

def build_module(cfg):
    if cfg == None or cfg == {}:
        return None

    if isinstance(cfg, list):
        return build_sequential(cfg)

    cls_ = module_obj[cfg['type']]
    del cfg['type']
    return cls_(**cfg)

def build_sequential(module_list):
    if module_list == None or module_list == {}:
        return None
        
    modules = []
    for cfg in module_list:
        modules.append(build_module(cfg))
    return nn.Sequential(*modules)

def build_decisions(decisions):
    if decisions == None or decisions == {}:
        return None

    emb = decisions['embedding']
    return nn.ModuleDict({k:build_module(copy.deepcopy(emb)) for k in decisions if k != 'embedding'})

def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

@FUSIONMODELS.register_module()
class ReIDNet(BaseDetector):
    def __init__(self,
                 losses_to_use,alpha,
                 backbone,numpoints,
                 cls_head,match_head,shape_head,fp_head,downsample,
                 cross_stage1,local_stage1,cross_stage2,local_stage2,
                 triplet_margin,triplet_p,triplet_sample_num,
                 output_feat_size,num_classes,use_o,eval_only=False,train_cfg=None,test_cfg=None,use_dgcnn=False):
                 
        super().__init__()

        self.losses_to_use = dict(kl=False,match=True,cls=False,fp=False)
        self.losses_to_use.update(losses_to_use)

        self.backbone = build_module(backbone)
        self.cls_head = build_module(cls_head)
        self.match_head = build_module(match_head)
        self.shape_head = build_module(shape_head)
        self.fp_head = build_module(fp_head)
        self.downsample = build_module(downsample)

        self.cross_stage1 = build_module(cross_stage1)
        self.cross_stage2 = build_module(cross_stage2)
        self.local_stage1 = build_module(local_stage1)
        self.local_stage2 = build_module(local_stage2)

        self.numpoints = numpoints

        self.maxpool = nn.MaxPool1d(output_feat_size)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(log_target=True,reduction='none')
        self.lsmx = nn.LogSoftmax(dim=1)
        self.smooth_l1 = nn.SmoothL1Loss(reduce=True,reduction='mean',beta=1.0)
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin,p=triplet_p)
        self.triplet_sample_num = triplet_sample_num
        
        self.alpha = alpha
        self.use_o = use_o
        self.eval_only = eval_only
        self.use_dgcnn = use_dgcnn

        self.verbose = False
        self.compute_summary = True

    ###########################################
    # Tracking Model
    ###########################################
    def tracking_train(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.cross_stage1.parameters():
            param.requires_grad = True
        for param in self.cross_stage2.parameters():
            param.requires_grad = True
        for param in self.match_head.parameters():
            param.requires_grad = True

    def tracking_eval(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward_inference(self, pts_batched):
        with torch.no_grad():
            return self.backbone(pts_batched, self.numpoints)

    ###########################################
    # ReID Subfunctions
    ###########################################
    def accum_val_log(self):
        match_gt = torch.cat(self.val_log['val_match_gt'],dim=0)
        match_preds = torch.cat(self.val_log['val_match_preds'],dim=0)
        match_acc = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match_gt).float().mean().item()

        cls_gt = torch.cat(self.val_log['val_cls_gt'],dim=0)
        cls_preds = torch.cat(self.val_log['val_cls_preds'],dim=0)
        cls_acc = cls_preds.argmax(dim=1).eq(torch.cat(cls_gt,dim=0)).float().mean().item()

        del self.val_log['val_match_gt']
        del self.val_log['val_match_preds']
        del self.val_log['val_cls_gt']
        del self.val_log['val_cls_preds']
        self.val_log.update({'val_match_acc':match_acc,'val_cls_acc':cls_acc})
        return self.val_log

    def xcorr(self, search_feat, search_xyz, template_feat, template_xyz):
        search_feat1_a = self.cross_stage1(search_feat, search_xyz, template_feat, template_xyz)
        search_feat1_b = self.local_stage1(search_feat1_a, search_xyz)
        search_feat2_a = self.cross_stage2(search_feat1_b, search_xyz, template_feat, template_xyz)
        search_feat2_b = self.local_stage2(search_feat2_a, search_xyz)

        return search_feat2_b
    
    def xcorr_eff(self, o1, xyz1, o2, xyz2):
        o1__ = self.cross_stage1(o1, xyz1, o2, xyz2)
        o2__ = self.cross_stage1(o2, xyz2, o1, xyz1)
        o1 = self.cross_stage2(o1__, xyz1, o2__, xyz2)
        o2 = self.cross_stage2(o2__, xyz2, o1__, xyz1)
        
        out = torch.cat([o1,o2],dim=2)
        
        return out, o1, o2

    def preprocess_inputs(self, sparse_1,sparse_2,label_1,label_2,id_1,id_2):
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        id_1 = torch.cat(id_1,dim=0)
        id_2 = torch.cat(id_2,dim=0)

        return sparse_1,sparse_2,label_1,label_2,id_1,id_2

    def preprocess_inputs_size_vis(self, sparse_1,sparse_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2):
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        id_1 = torch.cat(id_1,dim=0)
        id_2 = torch.cat(id_2,dim=0)
        size_1 = torch.cat(size_1,dim=0)
        size_2 = torch.cat(size_2,dim=0)
        vis_1 = torch.cat(vis_1,dim=0)
        vis_2 = torch.cat(vis_2,dim=0)
        
        return sparse_1,sparse_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2

    def siamese_forward(self,sparse_1,sparse_2):
        assert sparse_1.shape == sparse_2.shape
        b, num_points,_ = sparse_1.shape

        if self.use_dgcnn:
            xyz, h = self.backbone(torch.cat([sparse_1,sparse_2],dim=0).permute(0,2,1),self.numpoints)
            h = h.permute(0,2,1)
            h = h.reshape(-1,h.shape[-1])
            h = self.downsample(h).reshape(2*b,num_points,-1).permute(0,2,1)
            return xyz[:b,...].permute(0,2,1), xyz[b:,...].permute(0,2,1), h[:b,...], h[b:,...]
        else:
            xyz, h = self.backbone(torch.cat([sparse_1,sparse_2],dim=0),self.numpoints)

            return xyz[:b,...], xyz[b:,...], h[:b,...], h[b:,...]

    def get_match_supervision(self,h1,h2,xyz1,xyz2,id_1,id_2):
        return h1, h2, xyz1, xyz2, ( id_1 == id_2 ).float()

    def cls_forward(self,h,target,log_vars,device,prefix=''):
        if self.losses_to_use['cls']:
            input_ = self.get_pooled_feats(h)
            cls_preds = self.cls_head(input_).squeeze(1)
            cls_loss = self.ce(cls_preds,target)

            cls_loss = cls_loss * self.alpha['cls']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'cls_loss'] = cls_loss.item()
                log_vars[prefix+'cls_acc'] = cls_preds.argmax(dim=1).eq(target).float().mean().item()
        else:
            cls_preds = None
            cls_loss = torch.tensor(0.,requires_grad=True,device=device)

        return cls_preds, cls_loss

    def fp_forward(self,h,target,log_vars,device,prefix=''):
        if self.losses_to_use['fp']:
            input_ = self.get_pooled_feats(h)
            fp_preds = self.fp_head(input_).squeeze(1)
            target = ( target > 9 ).float()
            fp_loss = self.bce(fp_preds, target)
            fp_loss = fp_loss * self.alpha['fp']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'fp_loss'] = fp_loss.item()
                log_vars[prefix+'fp_acc'] = (nn.Sigmoid()(fp_preds) > 0.5).float().eq(target).float().mean().item()
        else:
            fp_preds = None
            fp_loss = torch.tensor(0.,requires_grad=True,device=device)

        return fp_preds, fp_loss

    def match_forward(self,h1,h2,xyz1,xyz2,match,log_vars,device,prefix=''):
        o1, o2 = None, None
        if self.losses_to_use['match']:
            match_in, o1, o2 = self.xcorr_eff(h1,xyz1,h2,xyz2)
            # match_in = self.xcorr(h1,xyz1,h2,xyz2)
            match_in = self.get_pooled_feats(match_in)
            match_preds = self.match_head(match_in).squeeze(1)
            match_loss = self.bce(match_preds,match)

            match_loss = match_loss * self.alpha['match']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'match_loss'] = match_loss.item()
                log_vars[prefix+'match_acc'] = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match).float().mean().item()

                gt_bins = torch.bincount(match.long())
                log_vars[prefix+'num_preds_0'] = gt_bins[0].item()
                log_vars[prefix+'num_preds_1'] = gt_bins[1].item() if len(gt_bins) > 1 else 0

                pred_bins = torch.bincount((nn.Sigmoid()(match_preds) > 0.5).long())
                log_vars[prefix+'num_gt_0'] = pred_bins[0].item()
                log_vars[prefix+'num_gt_1'] = pred_bins[1].item() if len(pred_bins) > 1 else 0
        else:
            match_preds = None
            match_loss = torch.tensor(0.,requires_grad=True,device=device)

        return match_preds, match_loss, (o1, o2)

    def match_forward_inference(self,h1,h2,xyz1,xyz2):
        match_in = self.xcorr(h1,xyz1,h2,xyz2)
        match_in = self.get_pooled_feats(match_in)
        match_preds = self.match_head(match_in).squeeze(1)

        return match_preds

    def get_kl_loss(self,h1,h2,match,log_vars,device,prefix=''):

        if self.losses_to_use['kl']:
            kl_loss = self.kl(self.lsmx(h1.reshape(h1.size(0),-1)),self.lsmx(h2.reshape(h2.size(0),-1))).mean(dim=1)
            where_no_match = torch.where(match == 0)
            kl_loss[where_no_match] = kl_loss[where_no_match] * -1
            kl_loss = kl_loss[match==0].mean() + kl_loss[match==1].mean()

            kl_loss = kl_loss * self.alpha['kl']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'kl_loss'] = kl_loss.item()
        else:
            kl_loss = torch.tensor(0.,requires_grad=True,device=device)

        return kl_loss

    def get_pooled_feats(self,h_cat):
        x1 = F.adaptive_max_pool1d(h_cat, 1).view(h_cat.size(0), -1)
        x2 = F.adaptive_avg_pool1d(h_cat, 1).view(h_cat.size(0), -1)
        return torch.cat((x1, x2), 1)
    
    def get_triplet_loss(self,h1,h2,id1,id2,match,log_vars,device,prefix=''):

        if self.losses_to_use['triplet']:
            match_idx = torch.where(match == 1)[0]
            matches = id1[match_idx]

            h_cat = torch.cat([h1,h2],dim=0)
            id_ = torch.cat([id1,id2],dim=0)

            a, p, n = [], [], []
            for i in range(matches.size(0)):


                m,idx = matches[i],match_idx[i]

                sample_pool = torch.where(id_ != m)[0]
                neg_idx_to_use = torch.multinomial(torch.ones(sample_pool.size(0)), self.triplet_sample_num, replacement=(len(sample_pool) < self.triplet_sample_num)).to(device)
                neg_idx_to_use = sample_pool[neg_idx_to_use]

                a.append(torch.full((self.triplet_sample_num,),idx,device=device))
                p.append(torch.full((self.triplet_sample_num,),idx,device=device))
                n.append(neg_idx_to_use)

            a = torch.cat(a,dim=0)
            p = torch.cat(p,dim=0)
            n = torch.cat(n,dim=0)

            a = h1.reshape(h1.size(0),-1)[a,...]
            p = h2.reshape(h1.size(0),-1)[p,...]
            n = h_cat.reshape(h_cat.size(0),-1)[n,...]


            triplet_loss = self.triplet_loss(anchor=a,
                                             positive=p,
                                             negative=n)
            triplet_loss = triplet_loss * self.alpha['triplet']


            if self.compute_summary and log_vars != None:
                log_vars[prefix+'triplet_loss'] = triplet_loss.item()
        else:
            triplet_loss = torch.tensor(0.,requires_grad=True,device=device)

        
        return triplet_loss

    ###########################################
    # ReID Model
    ###########################################
    def forward_train(self,sparse_1,sparse_2,label_1,label_2,id_1,id_2):
        if self.eval_only:
            exit(0)

        log_vars = {}
        losses = {}

        sparse_1,sparse_2,label_1,label_2,id_1,id_2 = self.preprocess_inputs(sparse_1,sparse_2,label_1,label_2,id_1,id_2)
        device = sparse_1.device

        xyz1, xyz2, h1, h2 = self.siamese_forward(sparse_1,sparse_2)

        # print("\033[91mxyz, h shape: \033[0m", xyz1.shape, h1.shape)

        # CLS Forward
        h_cat = torch.cat([h1,h2],dim=0)
        fp_filter = torch.where(torch.cat([id_1,id_2],dim=0) != -1)[0]
        cls_preds, cls_loss = self.cls_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        # FP Forward
        fp_preds, fp_loss = self.fp_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        # Match Forward
        h1, h2, xyz1, xyz2, match = self.get_match_supervision(h1,h2,xyz1,xyz2,id_1,id_2)
        match_preds, match_loss, (o1, o2) = self.match_forward(h1,h2,xyz1,xyz2,match,log_vars,device,prefix='')

        # KL Forward
        kl_loss = self.get_kl_loss(h1,h2,match,log_vars,device,prefix='')

        # Triplet Forward
        if self.use_o:
            h1, h2 = self.get_pooled_feats(o1), self.get_pooled_feats(o2)
        triplet_loss = self.get_triplet_loss(h1,h2,id_1,id_2,match,log_vars,device,prefix='')
        
        losses['reid_loss'] = match_loss + cls_loss + kl_loss + fp_loss + triplet_loss
        return losses, log_vars

    def forward_test(self,sparse_1,sparse_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2,*args,**kwargs):
        results = {}
        log_vars = None

        sparse_1,sparse_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2 = \
            self.preprocess_inputs_size_vis(sparse_1,sparse_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2)
        device = sparse_1.device

        fp_filter = torch.where(torch.cat([id_1,id_2],dim=0) != -1)[0]

        # Siamese Forward
        # xyz1, xyz2, h1, h2, fa1, fa2, fa3, fb1, fb2, fb3 = self.siamese_forward(sparse_1,sparse_2)
        xyz1, xyz2, h1, h2 = self.siamese_forward(sparse_1,sparse_2)
        h_cat = torch.cat([h1,h2],dim=0)

        # CLS Forward
        cls_preds, cls_loss = self.cls_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        # FP Forward
        fp_preds, fp_loss = self.fp_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        # Match Forward
        h1, h2, xyz1, xyz2, match = self.get_match_supervision(h1,h2,xyz1,xyz2,id_1,id_2)
        match_preds, match_loss, (o1,o2) = self.match_forward(h1,h2,xyz1,xyz2,match,log_vars,device,prefix='')

        
        ############
        # w1: save features by model & class
        # w2: save features by model (whole class as one)
        # w3: 
        # 0: car, 1: truck, 3: bus, 4: trailer, 6: motorcycle, 8: pedestrian, -1: unlabeled

        work = ["w1", "w2"]
        pca = PCA(n_components=3)
        model_name = "spotr"
        # class_to_extract = [0,1,3,4,6,8,-1]
        class_to_extract = [4]

        #################################################################################################################################################
        # Save Point Features by Model & Class
        #################################################################################################################################################

        for idx, match_pred in enumerate(match_preds):
            if int(label_1[idx]) in class_to_extract: # If [instance A] class in 'class_to_extract'
                fa_T = h1[idx].T
                fa_pca = pca.fit_transform(fa_T.cpu())

                if "w1" in work:
                    file_path = "./runs_feats/PCAFEAT_online/PCAFEAT_{}_{}.npy".format(model_name, int(label_1[idx]))
                    if os.path.exists(file_path):
                        loaded_stacked_feats = np.load(file_path)
                        print("A [Class: {}] [ID: {}] - {} / {}".format(int(label_1[idx]), int(id_1[idx]), loaded_stacked_feats.shape, fa_pca.shape))
                        updated_stacked_feats = np.append(loaded_stacked_feats, fa_pca, axis=0)
                        np.save(file_path, updated_stacked_feats)
                    else:
                        print("New Instance Feature_{}".format(int(label_1[idx])))
                        np.save(file_path, fa_pca)
                
                if "w2" in work:
                    file_path = "./runs_feats/PCAFEAT_online/PCAFEAT_{}_whole.npy".format(model_name)
                    if os.path.exists(file_path):
                        loaded_stacked_feats = np.load(file_path)
                        print("Stacking to whole")
                        updated_stacked_feats = np.append(loaded_stacked_feats, fa_pca, axis=0)
                        np.save(file_path, updated_stacked_feats)
                    else:
                        print("New Instance Feature_whole")
                        np.save(file_path, fa_pca)

            if int(label_2[idx]) in class_to_extract: # If [instance B] class in 'class_to_extract'
                fb_T = h2[idx].T
                fb_pca = pca.fit_transform(fb_T.cpu())

                if "w1" in work:
                    file_path = "./runs_feats/PCAFEAT_online/PCAFEAT_{}_{}.npy".format(model_name, int(label_2[idx]))
                    if os.path.exists(file_path):
                        loaded_stacked_feats = np.load(file_path)
                        print("B [Class: {}] [ID: {}] - {} / {}".format(int(label_2[idx]), int(id_2[idx]), loaded_stacked_feats.shape, fb_pca.shape))
                        updated_stacked_feats = np.append(loaded_stacked_feats, fb_pca, axis=0)
                        np.save(file_path, updated_stacked_feats)
                    else:
                        print("New Instance Feature_{}".format(int(label_1[idx])))
                        np.save(file_path, fb_pca)
                
                if "w2" in work:
                    file_path = "./runs_feats/PCAFEAT_online/PCAFEAT_{}_whole.npy".format(model_name)
                    if os.path.exists(file_path):
                        loaded_stacked_feats = np.load(file_path)
                        print("Stacking to whole")
                        updated_stacked_feats = np.append(loaded_stacked_feats, fb_pca, axis=0)
                        np.save(file_path, updated_stacked_feats)
                    else:
                        print("New Instance Feature_whole")
                        np.save(file_path, fb_pca)
        
        # KL Forward
        kl_loss = self.get_kl_loss(h1,h2,match,log_vars,device,prefix='')

        results['val_fp_loss'] = torch.tensor([fp_loss])
        results['val_match_loss'] = torch.tensor([match_loss])
        results['val_cls_loss'] = torch.tensor([cls_loss])
        results['val_kl_loss'] = torch.tensor([kl_loss])
        results['val_match_preds'] = match_preds
        results['val_match_gt'] = match
        results['val_cls_preds'] = cls_preds
        results['val_cls_gt'] = torch.cat([label_1,label_2],dim=0)
        results['val_fp_preds'] = fp_preds
        results['val_fp_gt'] = ( torch.cat([label_1,label_2],dim=0) > 9 ).float()
        results['match_classes'] = torch.cat([label_1.unsqueeze(1),label_2.unsqueeze(1)],dim=1)

        results['is_fp'] = torch.logical_or( (label_1 > 9) , (label_2 > 9) )
        results['num_points'] = torch.cat([size_1.unsqueeze(1),size_2.unsqueeze(1)],dim=1)
        results['val_vis_gt_all'] = torch.cat([vis_1.unsqueeze(1),vis_2.unsqueeze(1)],dim=1)

        return [ results ]

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def train_step(self, data, optimizer):
        if self.verbose:
            t1 = time.time()
            print("{} Starting train_step()".format(self.log_msg()))

        losses, log_vars_train = self(**data)
        loss, log_vars = self._parse_losses(losses)
        log_vars.update(log_vars_train)
        
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))
        if self.verbose:
            print("{} Ending train_step() after {}s".format(self.log_msg(),time.time()-t1))

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))

        return outputs

    def extract_feat(self,*args,**kwargs):
        raise NotImplementedError

    def show_result(self):
        raise NotImplementedError

    def aug_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def simple_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def init_weights(self):
        pass
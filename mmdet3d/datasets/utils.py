
import os
import mmcv
import math
import torch
import itertools
import random 

import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

from collections.abc import Sequence
from mmcv.runner import get_dist_info

from torch_cluster import fps
import torch.distributed as dist
import os.path as osp

from .selfattention import AttentionLayer
from mmdet3d.models import pointnet2_utils as pu2


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor | None: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data

class NuscenesDicts():
    def __init__(self):
        pass

    def get_ds_name_to_scene_token(self,nusc):
        import nuscenes.utils.splits as s
        name_to_scene_token = {x['name']:x['token'] for x in nusc.scene}
        ds_name_to_scene_token = {}
        for name,train,val in [('v1.0-trainval',s.train,s.val,),
                                ('v1.0-medium',s.medium_train,s.medium_val,),
                                ('v1.0-mini',s.mini_train,s.mini_val,),
                                ('v1.0-balanced-medium',s.balanced_medium_train,s.balanced_medium_val,)]:
            
            ds_name_to_scene_token[name] = {
                'train':{x:name_to_scene_token[x] for x in train},
                'val':  {x:name_to_scene_token[x] for x in val}
            }
        return ds_name_to_scene_token


    def get_instance_token_to_id(self,nusc):
        nusc.instance = sorted(nusc.instance, key=lambda x : x['token'])
        instance_token_to_id = {}
        for x in nusc.instance:
            instance_token_to_id[x['token']] = len(instance_token_to_id)
        return instance_token_to_id


    def get_scene_token_to_keyframes(self,nusc):
        scene_token_to_keyframes = dict()
        for scene in nusc.scene:
            sample = nusc.get('sample',scene['first_sample_token'])
            sample_data = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
            temp = [sample_data['is_key_frame']]
            while(sample_data['next'] != ''):
                sample_data = nusc.get('sample_data',sample_data['next'])
                temp.append(sample_data['is_key_frame'])
                
            scene_token_to_keyframes[scene['token']] = temp
        return scene_token_to_keyframes



    def get_sample_token_to_num(self,nusc):
        sample_token_to_num = dict()
        for scene in nusc.scene:
            sample = nusc.get('sample',scene['first_sample_token'])
            sample_data = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
            count = 0
            if sample_data['is_key_frame']:
                sample_token_to_num[sample_data['sample_token']] = count
            count += 1
            temp = [sample_data['is_key_frame']]
            while(sample_data['next'] != ''):
                sample_data = nusc.get('sample_data',sample_data['next'])
                if sample_data['is_key_frame']:
                    sample_token_to_num[sample_data['sample_token']] = count
                count += 1
                
        return sample_token_to_num


    def get_sample_to_scene(self,nusc):
        return {k['token']:k['scene_token'] for k in nusc.sample}

    def get_scene_to_sample(self,nusc):
        return {k['scene_token']:k['token'] for k in nusc.sample}

    def get_instance_to_scene(self,nusc):
        sample_to_scene = self.get_sample_to_scene(nusc)
        return {x['instance_token']:sample_to_scene[x['sample_token']] for x in nusc.sample_annotation}


    def get_sample_to_keyframes(self,nusc):
        sample_to_scene = self.get_sample_to_scene(nusc)
        scene_token_to_keyframes = self.get_scene_token_to_keyframes(nusc)
        return {sample:scene_token_to_keyframes[scene] for sample,scene in sample_to_scene.items()}


    def get_instance_to_keyframes(self,nusc):
        instance_to_scene = self.get_instance_to_scene(nusc)
        scene_token_to_keyframes = self.get_scene_token_to_keyframes(nusc)
        return {k:scene_token_to_keyframes[v] for k,v in instance_to_scene.items()}

def get_or_create_nuscenes_dict(filename,filepath='Datasets/NuScenes-ReID/data/nuscenes/nuscenes_dicts',nuscenes_dataroot='Datasets/NuScenes-ReID/data/nuscenes'):
    """Method to create or load a nuscenes dict from disk.

    This method allows for creating small files that contain dataset information at the begining of training.

    Args:
        filename (str): name of the file to create or load; also doubles as the suffix for the 
                        dict creation method.
        filepath (str): path to the directory where the file should be created or loaded.
        nuscenes_dataroot (str, optional): path to the nuscenes dataset. Defaults to 'data/nuscenes'.
    """
    assert filename.endswith('.pkl') or filename.endswith('.json'), 'name should end with .pkl or .json'

    print("Filename: ", filename)

    rank, world_size = get_dist_info()
    if not osp.isfile(osp.join(filepath,filename)) and rank == 0:
        from nuscenes import NuScenes
        # nusc =  NuScenes(dataroot="/home/data/nuscenes",version='v1.0-mini')
        # nusc =  NuScenes(dataroot="/home/data",version='lstk')
        # nusc =  NuScenes(dataroot=nuscenes_dataroot,version='v1.0-trainval')
        nusc =  NuScenes(dataroot=nuscenes_dataroot,version='v1.0-mini')
        out = getattr(NuscenesDicts(),'get_{}'.format(filename.split('.')[0]))(nusc)

        if not osp.isdir(filepath):
            os.makedirs(filepath)

        
        if filename.endswith('.pkl'):
            with open(osp.join(filepath,filename),'wb') as f:
                import pickle as pkl
                pkl.dump(out,f)
        elif filename.endswith('.json'):
            with open(osp.join(filepath,filename),'w') as f:
                import json
                json.dump(out,f)


    if world_size > 1:
        dist.barrier()

    if filename.endswith('.pkl'):
        with open(osp.join(filepath,filename),'rb') as f:
            import pickle as pkl
            return pkl.load(f)
    elif filename.endswith('.json'):
        with open(osp.join(filepath,filename),'r') as f:
            import json
            return json.load(f)

class MatchingEval(object):
    def __init__(self):
        super().__init__()
        pass

    def f1_precision_recall(self, preds, targets):
        log_vars = {}

        pos_idx = torch.where(targets == 1)[0]
        recall_pos = (preds[pos_idx]).sum() / (targets[pos_idx].sum() + 1e-6)
        precision_pos = (preds[pos_idx]).sum() / (preds.sum() + 1e-6)
        f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos + 1e-6)

        log_vars['val_match_f1_pos'] = f1_pos.item()
        log_vars['val_match_recall_pos'] = recall_pos.item()
        log_vars['val_match_precision_pos'] = precision_pos.item()

        neg_idx = torch.where(targets == 0)[0]
        recall_neg = (1 - preds[neg_idx]).sum() / (1 - targets[neg_idx]).sum() + 1e-6
        precision_neg = (1 - preds[neg_idx]).sum() / (1 - preds).sum() + 1e-6
        f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg + 1e-6)

        log_vars['val_match_f1_neg'] = f1_neg.item()
        log_vars['val_match_recall_neg'] = recall_neg.item()
        log_vars['val_match_precision_neg'] = precision_neg.item()

        return log_vars




    def evaluate_points(self,preds,targets,num_points):
        log_vars = {}
        preds = (nn.Sigmoid()(preds) > 0.5).float()


        unique_points = num_points.flatten().unique()
        max_ = num_points.max().item()
        min_ = num_points.min().item()

        h_exponent = np.log2(max_)

        buckets = [2**i for i in range(int(h_exponent)+1)]

        def filter_at_least_one(vis1,vis2,idx1,idx2):
            filtered_idx = torch.cat([vis1.unsqueeze(1),vis2.unsqueeze(1)],dim=1).max(1).values
            return torch.where(idx1 <= filtered_idx)

        def filter_at_least_both(vis1,vis2,idx1,idx2):
            filtered_idx = torch.cat([vis1.unsqueeze(1),vis2.unsqueeze(1)],dim=1).min(1).values
            return torch.where(idx1 <= filtered_idx)

        def only_of_a_pair(vis1,vis2,idx1_1,idx1_2,idx2_1,idx2_2):
            return torch.where(torch.logical_or(
                torch.logical_and(torch.logical_and(idx1_1 <= vis1, vis1 < idx1_2),
                                  torch.logical_and(idx2_1 <= vis2, vis2 < idx2_2),),
                torch.logical_and(torch.logical_and(idx2_1 <= vis1, vis1 < idx2_2),
                                  torch.logical_and(idx1_1 <= vis2, vis2 < idx1_2),),
            ))


        at_least_one = {}
        for i in range(len(buckets)-1):
            filter_ = filter_at_least_one(vis1=num_points[:,0],
                                          vis2=num_points[:,1],
                                          idx1=buckets[i],
                                          idx2=buckets[i+1])

            didx = (i,i+1,)
            at_least_one[didx] = self.f1_precision_recall(preds[filter_], targets[filter_])
            at_least_one[didx]['accuracy'] = (preds[filter_] == targets[filter_]).float().mean().item()
            at_least_one[didx]['num_observations_pos'] = len(torch.where( targets[filter_] == 1)[0])
            at_least_one[didx]['num_observations_neg'] = len(torch.where( targets[filter_] == 0)[0])

            for k,x in at_least_one[didx].items():
                if str(x).lower() == 'nan':
                    at_least_one[didx][k] = -1


        at_least_both = {}
        for i in range(len(buckets)-1):
            filter_ = filter_at_least_both(vis1=num_points[:,0],
                                           vis2=num_points[:,1],
                                           idx1=buckets[i],
                                           idx2=buckets[i+1])

            didx = (i,i+1,)
            at_least_both[didx] = self.f1_precision_recall(preds[filter_], targets[filter_])
            at_least_both[didx]['accuracy'] = (preds[filter_] == targets[filter_]).float().mean().item()
            at_least_both[didx]['num_observations_pos'] = len(torch.where( targets[filter_] == 1)[0])
            at_least_both[didx]['num_observations_neg'] = len(torch.where( targets[filter_] == 0)[0])


            for k,x in at_least_both[didx].items():
                if str(x).lower() == 'nan':
                    at_least_both[didx][k] = -1

        for_a_pair = {}
        for x in itertools.combinations_with_replacement([x for x in range(len(buckets)-1)],2):
            filter_ = only_of_a_pair(vis1=num_points[:,0],
                                     vis2=num_points[:,1],
                                     idx1_1=buckets[x[0]],
                                     idx1_2=buckets[x[0]+1],
                                     idx2_1=buckets[x[1]],
                                     idx2_2=buckets[x[1]+1])

            didx = ((x[0],x[0]+1,),(x[1],x[1]+1,),)
            for_a_pair[didx] = self.f1_precision_recall(preds[filter_], targets[filter_])
            for_a_pair[didx]['accuracy'] = (preds[filter_] == targets[filter_]).float().mean().item()
            for_a_pair[didx]['num_observations_pos'] = len(torch.where( targets[filter_] == 1)[0])
            for_a_pair[didx]['num_observations_neg'] = len(torch.where( targets[filter_] == 0)[0])


            for k,x in for_a_pair[didx].items():
                if str(x).lower() == 'nan':
                    for_a_pair[didx][k] = -1

        return dict(at_least_one=at_least_one,
                    at_least_both=at_least_both,
                    for_a_pair=for_a_pair)



    def evaluate_distance(self,preds,targets,num_points):
        log_vars = {}
        preds = (nn.Sigmoid()(preds) > 0.5).float()


        unique_points = num_points.flatten().unique()
        max_ = num_points.max().item()
        min_ = num_points.min().item()

        h_exponent = np.log2(max_+ 1e-6)

        buckets = [5*i for i in range(int(max_/5)+3)]

        def filter_at_least_one(vis1,vis2,idx1,idx2):
            filtered_idx = torch.cat([vis1.unsqueeze(1),vis2.unsqueeze(1)],dim=1).min(1).values
            return torch.where(filtered_idx <= idx1)

        def filter_at_least_both(vis1,vis2,idx1,idx2):
            filtered_idx = torch.cat([vis1.unsqueeze(1),vis2.unsqueeze(1)],dim=1).max(1).values
            return torch.where(filtered_idx <= idx1)

        def only_of_a_pair(vis1,vis2,idx1_1,idx1_2,idx2_1,idx2_2):
            return torch.where(torch.logical_or(
                torch.logical_and(torch.logical_and(idx1_1 <= vis1, vis1 < idx1_2),
                                  torch.logical_and(idx2_1 <= vis2, vis2 < idx2_2),),
                torch.logical_and(torch.logical_and(idx2_1 <= vis1, vis1 < idx2_2),
                                  torch.logical_and(idx1_1 <= vis2, vis2 < idx1_2),),
            ))


        at_least_one = {}
        for i in range(len(buckets)-1):
            filter_ = filter_at_least_one(vis1=num_points[:,0],
                                          vis2=num_points[:,1],
                                          idx1=buckets[i],
                                          idx2=buckets[i+1])

            didx = (i,i+1,)
            at_least_one[didx] = self.f1_precision_recall(preds[filter_], targets[filter_])
            at_least_one[didx]['accuracy'] = (preds[filter_] == targets[filter_]).float().mean().item()
            at_least_one[didx]['num_observations_pos'] = len(torch.where( targets[filter_] == 1)[0])
            at_least_one[didx]['num_observations_neg'] = len(torch.where( targets[filter_] == 0)[0])

            for k,x in at_least_one[didx].items():
                if str(x).lower() == 'nan':
                    at_least_one[didx][k] = -1


        at_least_both = {}
        for i in range(len(buckets)-1):
            filter_ = filter_at_least_both(vis1=num_points[:,0],
                                           vis2=num_points[:,1],
                                           idx1=buckets[i],
                                           idx2=buckets[i+1])

            didx = (i,i+1,)
            at_least_both[didx] = self.f1_precision_recall(preds[filter_], targets[filter_])
            at_least_both[didx]['accuracy'] = (preds[filter_] == targets[filter_]).float().mean().item()
            at_least_both[didx]['num_observations_pos'] = len(torch.where( targets[filter_] == 1)[0])
            at_least_both[didx]['num_observations_neg'] = len(torch.where( targets[filter_] == 0)[0])


            for k,x in at_least_both[didx].items():
                if str(x).lower() == 'nan':
                    at_least_both[didx][k] = -1

        for_a_pair = {}
        for x in itertools.combinations_with_replacement([x for x in range(len(buckets)-1)],2):
            filter_ = only_of_a_pair(vis1=num_points[:,0],
                                     vis2=num_points[:,1],
                                     idx1_1=buckets[x[0]],
                                     idx1_2=buckets[x[0]+1],
                                     idx2_1=buckets[x[1]],
                                     idx2_2=buckets[x[1]+1])

            didx = ((x[0],x[0]+1,),(x[1],x[1]+1,),)
            for_a_pair[didx] = self.f1_precision_recall(preds[filter_], targets[filter_])
            for_a_pair[didx]['accuracy'] = (preds[filter_] == targets[filter_]).float().mean().item()
            for_a_pair[didx]['num_observations_pos'] = len(torch.where( targets[filter_] == 1)[0])
            for_a_pair[didx]['num_observations_neg'] = len(torch.where( targets[filter_] == 0)[0])


            for k,x in for_a_pair[didx].items():
                if str(x).lower() == 'nan':
                    for_a_pair[didx][k] = -1

        return dict(at_least_one=at_least_one,
                    at_least_both=at_least_both,
                    for_a_pair=for_a_pair)


    def eval_per_visibility(self, preds, targets, vis_classes):
        log_vars = {}
        preds = (nn.Sigmoid()(preds) > 0.5).float()
        non_fp_preds = preds[targets != -1]
        non_fp_tragets = targets[targets != -1]
        non_fp_vis_classes = vis_classes[targets != -1]

        if len(non_fp_vis_classes.shape) == 3:
            non_fp_vis_classes = non_fp_vis_classes.squeeze(2)
    

        vis_levels = [0,1,2,3]

        def filter_at_least_one(vis1,vis2,tvis):
            return torch.where(torch.cat([vis1.unsqueeze(1),vis2.unsqueeze(1)],dim=1).max(1).values >= tvis)

        def filter_at_least_both(vis1,vis2,tvis):
            return torch.where(torch.cat([vis1.unsqueeze(1),vis2.unsqueeze(1)],dim=1).min(1).values >= tvis)

        def only_of_a_pair(vis1,vis2,tvis1,tvis2):
            return torch.where(torch.logical_or(
                torch.logical_and(vis1 == tvis1, vis2 == tvis2),
                torch.logical_and(vis1 == tvis2, vis2 == tvis1)
            ))

        at_least_one = {}
        for x in vis_levels:
            filter_ = filter_at_least_one(vis1=non_fp_vis_classes[:,0],
                                          vis2=non_fp_vis_classes[:,1],
                                          tvis=x)


            at_least_one[x] = self.f1_precision_recall(non_fp_preds[filter_], non_fp_tragets[filter_])
            at_least_one[x]['accuracy'] = (non_fp_preds[filter_] == non_fp_tragets[filter_]).float().mean().item()
            at_least_one[x]['num_observations_pos'] = len(torch.where( non_fp_tragets[filter_] == 1)[0])
            at_least_one[x]['num_observations_neg'] = len(torch.where( non_fp_tragets[filter_] == 0)[0])


        at_least_both = {}                        
        for x in vis_levels:
            filter_ = filter_at_least_both(vis1=non_fp_vis_classes[:,0],
                                        vis2=non_fp_vis_classes[:,1],
                                        tvis=x)

            at_least_both[x] = self.f1_precision_recall(non_fp_preds[filter_], non_fp_tragets[filter_])
            at_least_both[x]['accuracy'] = (non_fp_preds[filter_] == non_fp_tragets[filter_]).float().mean().item()
            at_least_both[x]['num_observations_pos'] = len(torch.where( non_fp_tragets[filter_] == 1)[0])
            at_least_both[x]['num_observations_neg'] = len(torch.where( non_fp_tragets[filter_] == 0)[0])

        for_a_pair = {}
        for x in itertools.combinations_with_replacement(vis_levels,2):
            filter_ = only_of_a_pair(vis1=non_fp_vis_classes[:,0],
                                    vis2=non_fp_vis_classes[:,1],
                                    tvis1=x[0],
                                    tvis2=x[1])

            for_a_pair[x] = self.f1_precision_recall(non_fp_preds[filter_], non_fp_tragets[filter_])
            for_a_pair[x]['accuracy'] = (non_fp_preds[filter_] == non_fp_tragets[filter_]).float().mean().item()
            for_a_pair[x]['num_observations_pos'] = len(torch.where( non_fp_tragets[filter_] == 1)[0])
            for_a_pair[x]['num_observations_neg'] = len(torch.where( non_fp_tragets[filter_] == 0)[0])


            for k,v in for_a_pair[x].items():
                if str(v).lower() == 'nan':
                    for_a_pair[x][k] = -1

        return dict(at_least_one=at_least_one,
                    at_least_both=at_least_both,
                    for_a_pair=for_a_pair)

def set_seeds(seed=0):
    print("setting seed",seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_tup_str(d):
    new = {}
    for k in d.keys():
        if type(k) == tuple:
            if type(d[k]) == dict:
                new[str(k)] = make_tup_str(d[k])
            else:
                new[str(k)] = d[k]
        else:
            if type(d[k]) == dict:
                new[k] = make_tup_str(d[k])
            else:
                new[k] = d[k]
    return new

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.float32):
        return torch.FloatTensor([data])
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data.astype(np.float32))
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif isinstance(data, np.int64):
        return torch.LongTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def get_knn(coi_to_size, k_num=200):
    print('Calculating KNN for K={}'.format(k_num))
    obj_id_to_knn = {}
    for k,v in coi_to_size.items():
        print("Processing ",k,'...')
        sizes = np.array([x[0] for x in v])
        ids = np.array([x[1] for x in v])
        dists = ((sizes[:, None, :] - sizes[None, :, :]) ** 2 ).sum(2)
        for i,obj_idx in enumerate(ids):
            sorted_idx = np.argsort(dists[i,:])
            obj_id_to_knn[obj_idx] = list( ids[
                sorted_idx[1:k_num+1] #only keep the first k and skip self
            ])
            
    return obj_id_to_knn

def subsamplePC(PC, subsample_number):

    subsample_option = 1
    # 1: original random
    # 2: greedy attention layer
    # 3: attention layer modified

    ################################################################# OPTION 1 ###################################
    if subsample_option == 1:
        if subsample_number == 0:
            pass
        elif PC.shape[1] > 2:
            if PC.shape[0] > 3:
                PC = PC[0:3, :]
                # print("HERE")
            if PC.shape[1] != subsample_number:
                # subsample
                new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=subsample_number, dtype=np.int64)
                PC = PC[:, new_pts_idx]
                # print("TTTTTHERE")
            PC = PC.reshape(3, subsample_number)
            # print("YOppppp")
        else:
            PC = np.zeros((3, subsample_number))
        
        return np.moveaxis(PC,1,0)
    ################################################################# OPTION 2 ###################################
    elif subsample_option == 2:
        if subsample_number == 2048:
            PC = np.zeros((3, subsample_number))
        if subsample_number == 2048:
            PC = np.zeros((3, subsample_number))
        elif PC.shape[1] > 2:
            if PC.shape[0] > 3:
                PC = PC[0:3, :]

            attention_layer = AttentionLayer(point_features_dim=PC.shape[0], heads=1)

            # Extract Attention Matrix
            xyz = np.reshape(PC, (1, PC.shape[1], PC.shape[0]))
            out, attention = attention_layer(torch.from_numpy(xyz))

            attention = attention.reshape(attention.shape[2], attention.shape[3])
            # Importance Score Array
            importance_scores = torch.sum(attention, dim=0)

            # Rank the points
            sorted_indices = torch.argsort(importance_scores, descending=True)

            # Select Subsamples from original indices
            subsampled_indices = []
            if PC.shape[1] >= subsample_number:
                subsampled_indices = sorted_indices[:subsample_number]
                PC = PC[:, subsampled_indices]
            else:
                duplicate_ratio = math.floor(subsample_number / PC.shape[1])
                PC_duplicated = np.tile(PC, (1, duplicate_ratio))

                remain_count = subsample_number - PC_duplicated.shape[1]
                if remain_count > 0:
                    new_cols = PC[:, sorted_indices[:remain_count]]
                    if new_cols.ndim == 1:
                        new_cols = new_cols[:, np.newaxis]  # or arr = arr[:, None]
                    try:
                        PC_duplicated = np.append(PC_duplicated, new_cols, axis=1)
                    except ValueError as e:
                        print(np.shape(PC_duplicated), np.shape(new_cols))
                else:
                    pass

                PC = PC_duplicated
        else:
            PC = np.zeros((3, subsample_number))
            
        return np.moveaxis(PC,1,0)
    ################################################################# OPTION 3 ###################################
    elif subsample_option == 3:
        if subsample_number == 2048:
            PC = np.zeros((3, subsample_number))
        if subsample_number == 2048:
            PC = np.zeros((3, subsample_number))
        elif PC.shape[1] > 2:
            if PC.shape[0] > 3:
                PC = PC[0:3, :]

            attention_layer = pu2.Self_Attention(d_model=32, nhead=2, attention='linear')

            mlp=[3, 32, 32, 32]
            last_channel = mlp[0]

            mlp_convs = nn.ModuleList()
            mlp_bns = nn.ModuleList()

            for out_channel in mlp[1:]:
                mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

            xyz = np.reshape(PC, (1, PC.shape[1], PC.shape[0]))
            new_xyz, new_points = pu2.sample_and_group_edge(npoint=None, radius=0.3, nsample=32, xyz=torch.from_numpy(xyz), points=None, sampling="FULL", numpoints=128, returnfps=False, use_knn=True)
            new_points = new_points.permute(0, 3, 1, 2)     # [B, D, numpoints, nsample]
            for i, conv in enumerate(mlp_convs):
                bn = mlp_bns[i]
                new_points =  F.relu(bn(conv(new_points)))

            new_points = torch.max(new_points, 3)[0]        # B x D x numpoints

            point_weights = attention_layer.forward_subsample(feat=new_points,xyz=new_xyz, mask=None).squeeze(0)
            sorted_indices = torch.argsort(point_weights, descending=True)
            # sorted_indices = sorted_indices.view()

            # print("point weights: ", point_weights.shape)
            # print("sorted idx: ", sorted_indices)

            # Select Subsamples from original indices
            subsampled_indices = []
            if PC.shape[1] >= subsample_number:
                subsampled_indices = sorted_indices[:subsample_number]
                PC = PC[:, subsampled_indices]
            else:
                duplicate_ratio = math.floor(subsample_number / PC.shape[1])
                PC_duplicated = np.tile(PC, (1, duplicate_ratio))

                remain_count = subsample_number - PC_duplicated.shape[1]
                if remain_count > 0:
                    # print("rc: ", remain_count)
                    # print("sorted_indices[:remain_count]: ", sorted_indices[:remain_count])
                    new_cols = PC[:, sorted_indices[:remain_count]]
                    # print("nc shape: ", new_cols.shape)
                    if new_cols.ndim == 1:
                        new_cols = new_cols[:, np.newaxis]  # or arr = arr[:, None]
                    try:
                        PC_duplicated = np.append(PC_duplicated, new_cols, axis=1)
                    except ValueError as e:
                        print("ERROR: ", np.shape(PC_duplicated), np.shape(new_cols))
                else:
                    pass

                PC = PC_duplicated
        
        else:
            PC = np.zeros((3, subsample_number))
        
        return np.moveaxis(PC,1,0)

def subsample_and_fill_PC(PC, subsample_number, fill_number):

    if subsample_number == 0:
        pass
    elif PC.shape[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != subsample_number:
            # subsample
            new_pts_idx = np.random.randint(low=0, high=PC.shape[1], size=subsample_number, dtype=np.int64)
            if fill_number > subsample_number:
                fill_pts_idx = np.random.randint(low=0, high=subsample_number, size=fill_number - subsample_number, dtype=np.int64)
                fill_pts_idx = new_pts_idx[fill_pts_idx]
                new_pts_idx = np.concatenate((new_pts_idx, fill_pts_idx), axis=0)

            PC = PC[:, new_pts_idx]
        PC = PC.reshape(3, fill_number)
    else:
        PC = np.zeros((3, fill_number))
    
    return np.moveaxis(PC,1,0)

def fps_or_interpolate(xyz, npoint):

    N, C = np.shape(xyz)

    if N > npoint:
        points_left = np.arange(len(xyz))
        sample_inds = np.zeros(npoint, dtype='int')
        dists = np.ones_like(points_left) * float('inf')

        selected = np.random.randint(N)
        sample_inds[0] = points_left[selected]

        points_left = np.delete(points_left, selected) # [P - 1]

        for i in range(1, npoint):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]
            
            dist_to_last_added_point = ((xyz[last_added] - xyz[points_left])**2).sum(-1) # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left]) # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        mask = np.zeros(xyz.shape[0], dtype=bool)
        mask[sample_inds] = True

        return xyz[mask]
    elif N == npoint:
        return xyz
    else:
        raise TypeError("subsample_sparse must be <= than min_points")

    # Deprecated
    # """
    # Input:
    #     xyz: pointcloud frame, [N, 3]
    #     npoint: number of samples
    # Return:
    #     centroids: sampled pointcloud index, [B, npoint]
    # """
    # # device = xyz.device
    # centroids = torch.zeros(npoint, dtype=torch.long)
    # distance = torch.ones(N) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long)
    # batch_indices = torch.arange(B, dtype=torch.long)
    # for i in range(npoint):
    #     centroids[:, i] = farthest
    #     centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
    #     dist = torch.sum((xyz - centroid) ** 2, -1)
    #     mask = dist < distance
    #     distance[mask] = dist[mask]
    #     farthest = torch.max(distance, -1)[1]
    # return centroids

def rand_crop(xyz, npoint):
    N, C = np.shape(xyz)

    if N > npoint:
        # Step 1: Pick a random point
        start_index = np.random.randint(0, N)
        current_point = xyz[start_index]

        # Step 2: Initialize the new array with the selected point
        ordered_points = np.empty_like(xyz)
        ordered_points[0] = current_point

        # Mask to keep track of which points have been added
        mask = np.ones(N, dtype=bool)
        mask[start_index] = False

        # Step 3: Iterate to fill the rest of the ordered points
        for i in range(1, N):
            remaining_points = xyz[mask]
            distances = np.linalg.norm(remaining_points - current_point, axis=1)
            next_point_index = np.argmin(distances)
            current_point = remaining_points[next_point_index]
            ordered_points[i] = current_point
            
            # Update mask
            global_index = np.nonzero(mask)[0][next_point_index]
            mask[global_index] = False

        # Check Complete
        # indices = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        # ordered_xyz = xyz[indices]

        # indices = np.lexsort((ordered_points[:, 2], ordered_points[:, 1], ordered_points[:, 0]))
        # ordered_ordered_points = ordered_points[indices]

        # print("\033[91mCheck Proper?: \033[0m", np.array_equal(ordered_xyz, ordered_ordered_points))

        return ordered_points[-npoint:]
    elif N == npoint:
        return xyz
    else:
        raise TypeError("subsample_sparse must be <= than min_points")
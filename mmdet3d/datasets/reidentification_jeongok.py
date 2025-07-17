import os
import json
import itertools
import numpy as np

from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from .reidentification_base import ReIDDatasetBase
from .utils import (set_seeds, make_tup_str, to_tensor, \
                    subsamplePC, get_or_create_nuscenes_dict)

def random_rotate(pc, roll_range=(-np.pi/4, np.pi/4), pitch_range=(-np.pi/4, np.pi/4), yaw_range=(0, 2*np.pi)):
    # Roll (X axis)
    roll = np.random.uniform(*roll_range)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    # Pitch (Y axis)
    pitch = np.random.uniform(*pitch_range)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # Yaw (Z axis)
    yaw = np.random.uniform(*yaw_range)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Apply in order: roll, pitch, yaw
    R = Rz @ Ry @ Rx
    return pc @ R.T

def random_scale(pc, scale_range=(0.95, 1.05)):
    scale = np.random.uniform(*scale_range)
    return pc * scale

def random_translate(pc, shift_range=0.2):
    shift = np.random.uniform(-shift_range, shift_range, 3)
    return pc + shift

def random_jitter(pc, sigma=0.01, clip=0.05):
    jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
    return pc + jitter

AUGMENTATION_FUNCS = {
    "rotate": random_rotate,
    "scale": random_scale,
    "translate": random_translate,
    "jitter": random_jitter,
}

@DATASETS.register_module()
class ReIDDatasetJeongok(ReIDDatasetBase):
    def __init__(self, *args, num_pairs_per_object=100, augmentations=None, **kwargs):
        self.num_pairs_per_object = num_pairs_per_object
        self.augmentations = augmentations if augmentations is not None else []
        super().__init__(*args, **kwargs)
        
        self.instance_token_to_id = dict()

        # Get base_path from config instead of hardcoding
        # The sparse_loader will be initialized in the parent class
        # We'll get the path from the sparse_loader's data_root
        super().__init__(*args, **kwargs)
        
        # Now that sparse_loader is initialized, get the base_path from it
        base_path = os.path.join(self.sparse_loader.data_root, 'objects')
        id = 0
        for instance_token in os.listdir(base_path):
            if instance_token.startswith('FP'):
                pass
            else:
                self.instance_token_to_id[instance_token] = id
                id += 1

        self.obj_tokens = list(self.sparse_loader.obj_id_to_nums.keys())

        self.collect_dataset_idx()

        self.maintain_api()

        self.vis_to_cls_id = {1:0,2:1,3:2,4:3}

    def before_collect_dataset_idx_hook(self):
        pass

    def after_collect_dataset_idx_hook(self):
        pass

    def __len__(self):
        return len(self.idx) * self.num_pairs_per_object

    def apply_augmentations(self, pc):
        for aug in self.augmentations:
            if aug in AUGMENTATION_FUNCS:
                pc = AUGMENTATION_FUNCS[aug](pc)
        return pc

    def __getitem__(self, i):
        obj_idx = i % len(self.idx)
        l1 = self.classes[obj_idx]
        pos_obj_tok = self.obj_tokens[obj_idx]
        # Only assign id1 for true positives
        id1 = self.instance_token_to_id.get(pos_obj_tok, -1)
        
        
        if np.random.choice([0,1]) == 1:
            pos_choices = self.get_random_frame(pos_obj_tok,2,replace=False)
            s1 = self.sparse_loader[(pos_obj_tok,pos_choices[0],)]
            s2 = self.sparse_loader[(pos_obj_tok,pos_choices[1],)]
            # Apply augmentations during training
            if getattr(self, 'training', True):
                s1 = self.apply_augmentations(s1)
                s2 = self.apply_augmentations(s2)
            return self.return_item(s1,s2,l1,l1,id1,id1)
        else:
            pos_choice = self.get_random_frame(pos_obj_tok,1,replace=False)[0]
            s1 = self.sparse_loader[(pos_obj_tok,pos_choice,)]
            
            neg_obj_tok, l2, density = self.get_random_other_even_train(taken_idx=obj_idx,
                                                                        taken_cls=l1,
                                                                        distribution=self.sparse_loader.obj_infos[pos_obj_tok]['distribution'])
            
            if neg_obj_tok.startswith("FP"):
                id2 = -1
            else:
                id2 = self.instance_token_to_id[neg_obj_tok]
            
            neg_choice = self.sparse_loader.get_random_frame_even(neg_obj_tok,1,density=density,replace=False)[0]
            s2 = self.sparse_loader[(neg_obj_tok,neg_choice,)]
            # Apply augmentations during training
            if getattr(self, 'training', True):
                s1 = self.apply_augmentations(s1)
                s2 = self.apply_augmentations(s2)
            return self.return_item(s1,s2,l1,l2,id1,id2)
        
@DATASETS.register_module()
class ReIDDatasetJeongokValEven(ReIDDatasetJeongok):

    def __init__(self,max_combinations,test_mode,*args,**kwargs):
        self.max_combinations = max_combinations
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.val_index)

    def __getitem__(self,idx):
        if idx < len(self.val_positives):
            sample = self.val_positives[idx]
            pos_obj_tok = sample['tok']
            s1 = self.sparse_loader[(pos_obj_tok,sample['o1'],)]
            s2 = self.sparse_loader[(pos_obj_tok,sample['o2'],)]

            l1 = sample['cls']

            id1 = self.instance_token_to_id[pos_obj_tok]

            v1 = self.sparse_loader.obj_infos[pos_obj_tok]['nums_to_distance'].get(int(sample['o1']),-1)
            v1 = 0#np.sqrt((self.sparse_loader.obj_infos[pos_obj_tok]['all_sizes'][v1,:2] ** 2).sum())
            v2 = self.sparse_loader.obj_infos[pos_obj_tok]['nums_to_distance'].get(int(sample['o2']),-1)
            v2 = 0#np.sqrt((self.sparse_loader.obj_infos[pos_obj_tok]['all_sizes'][v2,:2] ** 2).sum())

            return self.return_item_size_dist(s1,s2,l1,l1,id1,id1,v1,v2)
        else:
            idx = idx - len(self.val_positives)
            sample = self.val_negatives[idx]
            s1 = self.sparse_loader[(sample['tok1'],sample['o1'],)]
            s2 = self.sparse_loader[(sample['tok2'],sample['o2'],)]

            l1 = sample['cls1']
            l2 = sample['cls2']

            if sample['tok2'].startswith("FP"):
                id2 = -1
            else:
                id2 = self.instance_token_to_id[sample['tok2']]

            id1 = self.instance_token_to_id[sample['tok1']]

            v1 = self.sparse_loader.obj_infos[sample['tok1']]['nums_to_distance'].get(int(sample['o1']),-1)
            v1 = 0#np.sqrt((self.sparse_loader.obj_infos[sample['tok1']]['all_sizes'][v1,:2] ** 2).sum())
            v2 = self.sparse_loader.obj_infos[sample['tok2']]['nums_to_distance'].get(int(sample['o2']),-1)
            v2 = 0#np.sqrt((self.sparse_loader.obj_infos[sample['tok2']]['all_sizes'][v2,:2] ** 2).sum())
            
            return self.return_item_size_dist(s1,s2,l1,l2,id1,id2,v1,v2)

    def before_collect_dataset_idx_hook(self):
        set_seeds(seed=self.validation_seed)
        
    def after_collect_dataset_idx_hook(self):
        val_positives = []
        for i,c in zip(self.idx,self.classes):
            tok = self.obj_tokens[i]
            nums = self.sparse_loader.obj_id_to_nums[tok]
            combs = list(itertools.combinations(nums, r=2))
            np.random.shuffle(combs)
            combs=combs[:self.max_combinations] # dont use all combinations
            val_positives.extend([dict(o1=x[0],
                                       o2=x[1],
                                       pts1=self.sparse_loader.obj_infos[tok]['num_pts'][x[0]],
                                       pts2=self.sparse_loader.obj_infos[tok]['num_pts'][x[1]],
                                       tok=tok,
                                       cls=c) 
                                    for x in combs])
        self.val_positives = val_positives

        self.sparse_loader.get_buckets(self.idx.tolist()+self.false_positive_idx.tolist())
        self.fp_buckets = self.sparse_loader.get_all_buckets(self.false_positive_idx.tolist())

        # print(self.fp_buckets)

        fp_buckets_filtered = dict()
        temp_dict = dict()
        fp_exist = False
        for class_name, bucket in self.fp_buckets.items():
            for dimension, data_list in bucket.items():
                if len(data_list) >= 2:
                    temp_dict[dimension] = data_list
                    fp_exist = True
            fp_buckets_filtered[class_name] = temp_dict

        # print(fp_buckets_filtered)

        # print(fp_exist)
        if fp_exist:
            self.fp_buckets = fp_buckets_filtered
        else:
            self.fp_buckets = None

        # print(self.fp_buckets)

        self.tp_buckets = self.sparse_loader.get_all_buckets(self.idx.tolist())

        val_negatives = []
        for x in self.val_positives:
            other_token, cls2, other_choice = self.get_random_other_even_val(taken_idx=x['o1'],
                                                                         taken_cls=x['cls'],
                                                                         pts=x['pts2'],
                                                                         i=self.obj_tokens.index(x['tok']))

            # other_choice = self.get_random_frame_even(other_token,1,replace=False)[0]
            val_negatives.append(dict(o1=x['o1'],
                                      o2=other_choice,
                                      tok1=x['tok'],
                                      tok2=other_token,
                                      cls1=x['cls'],
                                      cls2=cls2))

        self.val_negatives = val_negatives
        self.val_index = np.arange(0,2*len(val_positives))
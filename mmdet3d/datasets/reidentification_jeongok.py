import os
import json
import itertools
import numpy as np

from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from .reidentification_base import ReIDDatasetBase
from .utils import (set_seeds, make_tup_str, to_tensor, \
                    subsamplePC, get_or_create_nuscenes_dict)

@DATASETS.register_module()
class ReIDDatasetJeongok(ReIDDatasetBase):
    def __init__(self,*args,**kwargs):
        
        self.instance_token_to_id = dict()

        base_path = 'Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/objects'
        id = 0
        for instance_token in os.listdir(base_path):
            if instance_token.startswith('FP'):
                pass
            else:
                self.instance_token_to_id[instance_token] = id
                id += 1

        super().__init__(*args, **kwargs)

        self.obj_tokens = list(self.sparse_loader.obj_id_to_nums.keys())

        self.collect_dataset_idx()

        self.maintain_api()

        self.vis_to_cls_id = {1:0,2:1,3:2,4:3}

    def before_collect_dataset_idx_hook(self):
        pass

    def after_collect_dataset_idx_hook(self):
        pass

    def __getitem__(self,idx):
        # idx --> index into self.idx
        # obj_idx --> corresponding index into valid obj_tokens
        
        pos_obj_idx = self.idx[idx]
        l1 = self.classes[idx]

        pos_obj_tok = self.obj_tokens[pos_obj_idx]

        id1 = self.instance_token_to_id[pos_obj_tok]
        
        
        if np.random.choice([0,1]) == 1:
            pos_choices = self.get_random_frame(pos_obj_tok,2,replace=False)
            s1 = self.sparse_loader[(pos_obj_tok,pos_choices[0],)]
            s2 = self.sparse_loader[(pos_obj_tok,pos_choices[1],)]

            return self.return_item(s1,s2,l1,l1,id1,id1)
        else:
            pos_choice = self.get_random_frame(pos_obj_tok,1,replace=False)[0]
            s1 = self.sparse_loader[(pos_obj_tok,pos_choice,)]
            
            neg_obj_tok, l2, density = self.get_random_other_even_train(taken_idx=pos_obj_idx,
                                                                        taken_cls=l1,
                                                                        distribution=self.sparse_loader.obj_infos[pos_obj_tok]['distribution'])
            
            if neg_obj_tok.startswith("FP"):
                id2 = -1
            else:
                id2 = self.instance_token_to_id[neg_obj_tok]
            
            neg_choice = self.sparse_loader.get_random_frame_even(neg_obj_tok,1,density=density,replace=False)[0]
            s2 = self.sparse_loader[(neg_obj_tok,neg_choice,)]

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

        print(self.fp_buckets)

        fp_buckets_filtered = dict()
        temp_dict = dict()
        fp_exist = False
        for class_name, bucket in self.fp_buckets.items():
            for dimension, data_list in bucket.items():
                if len(data_list) >= 2:
                    temp_dict[dimension] = data_list
                    fp_exist = True
            fp_buckets_filtered[class_name] = temp_dict

        print(fp_buckets_filtered)

        print(fp_exist)
        if fp_exist:
            self.fp_buckets = fp_buckets_filtered
        else:
            self.fp_buckets = None

        print(self.fp_buckets)

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
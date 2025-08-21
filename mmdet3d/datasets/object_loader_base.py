import os
import time

import os.path as osp
import numpy as np
import pickle as pkl
from itertools import islice

from mmdet.datasets import DATASETS
from lamtk.aggregation.loader import Loader
from lamtk.aggregation.utils import filter_metadata_by_scene_ids, combine_metadata
from mmdet3d.datasets.utils import get_or_create_generic_dict

from pathlib import Path

def combine_metadata_fix(metadata_path_sparse):
    """Combines metadata from multiple files into one dictionary, and fixes the 
    path to the data for files store under new_datasets."""

    md_files = os.listdir(metadata_path_sparse)
    sparse_metadata = {x:pkl.load(open(os.path.join(metadata_path_sparse,x),'rb')) \
                                            for x in md_files}

    for x in md_files:
        temp = [int(y) for y in x[-11:-4].split("-")]
        if 500 <= temp[0] and temp[1] <= 700:
            print(x)
            for k in sparse_metadata[x]['obj_infos']:
                sparse_metadata[x]['obj_infos'][k]['path'] = Path("../../../new_datasets/lstk/sparse-waymo-det-both-train") \
                                                             / sparse_metadata[x]['obj_infos'][k]['path']


    metadata = dict(scene_infos={}, obj_infos={}, frame_infos={})
    for metadata_i in sparse_metadata.values():

        metadata['scene_infos'].update(metadata_i['scene_infos'])
        metadata['obj_infos'].update(metadata_i['obj_infos'])
        metadata['frame_infos'].update(metadata_i['frame_infos'])
    return metadata

def load_metadata(metadata_path,use_fix=False):
    if metadata_path.endswith('.pkl'):
            with open(metadata_path, 'rb') as f:
                sparse_metadata = pkl.load(f)
    elif use_fix == True:
        sparse_metadata = combine_metadata_fix(metadata_path)
    else:
         sparse_metadata = combine_metadata([pkl.load(open(os.path.join(metadata_path,x),'rb')) \
                                                for x in os.listdir(metadata_path)])
         
    return sparse_metadata

def load_metadata_split(metadata_path,version,train):
    """Loads a specific split of the dataset.
    
    Args:
        metadata_path (str): Path to the metadata file.
        version (str): Version of the dataset to load.
        train (bool): Whether to load the training or validation split.
        use_metadata_fix (bool): Whether to use the metadata fix(only relevant to ).
    """

    splits_by_scene = get_or_create_generic_dict(filename='ds_name_to_scene_token.pkl',
                                                        filepath='Datasets/NuScenes-ReID/data/lstk',
                                                        generic_dataroot='Datasets/NuScenes-ReID/data/genericdataset')

    metadata = load_metadata(metadata_path,use_fix=False)
    split = splits_by_scene[version]['train'].values() if train else splits_by_scene[version]['val'].values()

    return filter_metadata_by_scene_ids(metadata,split)

def load_jeongok_metadata(metadata_path, train):
    """Load Jeongok metadata and apply train/val split.
    
    Args:
        metadata_path (str): Path to the metadata.pkl file
        train (bool): Whether to load training or validation split
    """
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pkl.load(f)
    except FileNotFoundError:
        print(f"Warning: Metadata file not found at {metadata_path}")
        return dict(scene_infos={}, obj_infos={}, frame_infos={})
    
    # Apply Jeongok-specific train/val split
    all_targets = list(metadata['obj_infos'].keys())
    
    # Separate regular targets from FP targets
    regular_targets = [t for t in all_targets if not t.startswith('FP')]
    fp_targets = [t for t in all_targets if t.startswith('FP')]
    
    # Sort regular targets numerically (target1, target2, etc.)
    regular_targets.sort(key=lambda x: int(x.replace('target', '')))
    
    # Use first 6 regular targets for training, last 2 for validation
    train_targets = regular_targets[:6]  # target1-target6
    val_targets = regular_targets[6:8]   # target7-target8
    # FP targets are always included in both splits
    
    if train:
        # For training, include train targets + FP targets
        valid_targets = train_targets + fp_targets
    else:
        # For validation, include val targets + FP targets  
        valid_targets = val_targets + fp_targets
    
    # Filter metadata to only include valid targets for this split
    filtered_metadata = {
        'scene_infos': metadata['scene_infos'],
        'obj_infos': {k: v for k, v in metadata['obj_infos'].items() if k in valid_targets},
        'frame_infos': {k: v for k, v in metadata['frame_infos'].items() 
                       if any(target in k for target in valid_targets)}
    }
    
    # print(f"[load_jeongok_metadata] {'Train' if train else 'Val'} split: {len(filtered_metadata['obj_infos'])} targets")
    return filtered_metadata

@DATASETS.register_module()
class ObjectLoaderSparseBase(Loader):

    def __init__(self, tracking_classes, min_points, **kwargs) -> None:
        super().__init__(**kwargs)

        self.min_points = min_points
        self.tracking_classes = tracking_classes

    def load(self, *args, **kwargs):
        raise NotImplementedError("This method shouw be implemented in the child class")
    
    def __getitem__(self, x):
        obj_id, frame_id = x
        obj = self.obj_infos.get(obj_id, None)
        if obj:
            return self.load(obj,str(frame_id))
        else:
            raise ValueError(f'obj_id {obj_id} does not exist in obj_infos')

    def get_filtered_nums(self,obj_key,obj_entry,min_points,only_keyframes=False):

        nums = sorted(list(obj_entry['num_pts'].keys()),key=lambda x : int(x))
        temp = np.array([obj_entry['num_pts'][int(x)] for x in nums])
        obj_filter = np.where(temp >= min_points)

        t = np.where(temp == 0)

        nums = np.array(nums)[obj_filter]
        temp = temp[obj_filter]

        nums_to_distance = dict()
        for i,num in enumerate(obj_entry['num_pts'].keys()):
            nums_to_distance[num] = i
        self.obj_infos[obj_key]['nums_to_distance'] = nums_to_distance

        return nums

    def collect_obj_id_to_nums(self,min_points,only_keyframes=False):
        obj_id_to_nums = {}
        for k,v in self.obj_infos.items():
            obj_id_to_nums[k] = self.get_filtered_nums(k,v,min_points,only_keyframes)
        return obj_id_to_nums
        
    def get_random_frame(self,obj_tok,num_samples,replace=False):
        nums = self.obj_id_to_nums[obj_tok]
        assert len(nums) >= num_samples, f"Assertion Error nums {nums} encountered for obj_tok:{obj_tok}"
        return np.random.choice(nums,num_samples,replace=replace)
        
    def get_buckets(self,index):
        """This function is used to calculated the distribution of the number of 
            points in each object."""
        self.buckets = [(2**x,2**(x+1)) for x in range(20)]
        temp = list(self.obj_id_to_nums.keys())
        for idx in index:
            obj = self.obj_infos[temp[idx]]
            obj_buckets = {}
            for n in self.obj_id_to_nums[obj['id']]:
                npts = obj['num_pts'][n]
                key = self.buckets[int(self.special_log(npts))]
                try:
                    obj_buckets[key] += [n]
                except KeyError:
                    obj_buckets[key] = [n]
            obj['buckets'] = obj_buckets
            obj['distribution'] = np.array([len(obj['buckets'].get(x,[])) for x in self.buckets])
            obj['distribution'] = obj['distribution']/obj['distribution'].sum()

    def get_all_buckets(self,index):
        """This function is used to accumulated the distribution of the power two 
        buckets over all the objects
        """
        all_buckets = {}
        temp = list(self.obj_id_to_nums.keys())
        for idx in index:
            obj = self.obj_infos[temp[idx]]
            cls_ = self.tracking_classes.get(obj['class_name'],None)
            if cls_ is None:
                #skip non tracking classes
                continue

            # if obj['id'].startswith('FP_'):
            #     cls_ = 'FP_' + cls_temp
            # else:
            #     cls_ = cls_temp

            all_buckets[cls_] = all_buckets.get(cls_,{})
            for k,pts_list in obj['buckets'].items():
                try:
                    all_buckets[cls_][k].append((obj['id'],len(pts_list),))
                except KeyError:
                    all_buckets[cls_][k] = [(obj['id'],len(pts_list),)]

        self.all_buckets = all_buckets
        return self.all_buckets

    def get_random_frame_even(self,obj_tok,num_samples,density,replace=False):
        """Finds and observation of a given object that has a certain density."""
        obj_buckets = self.obj_infos[obj_tok]['buckets']
        while(len(obj_buckets.get(self.buckets[density],[])) == 0):
            density -= 1
            if density == -1:
                density = 0 
                while(len(obj_buckets.get(self.buckets[density],[])) == 0):
                    density += 1
                    if density >= len(self.buckets):
                        raise Exception("Infinite loop will occur in get_random_frame_even()")
        nums = obj_buckets.get(self.buckets[density],[])
        return np.random.choice(nums,num_samples,replace=replace)

    def get_class_list_density(self,class_name,density_idx):
        """Finds a density bucket that has at least 2 objects in it.
        
        The density bucket is first found by going down the list of buckets until a bucket 
        with at least 2 objects is found. If no such bucket is found, the function goes up the list 
        of buckets until a bucket with at least 2 objects is found. If no such bucket is found, 
        an exception is raised.
    
        Arguments:
            class_name {str} -- The class name of the object
            density_idx {int} -- The index of the density bucket to start searching from
        """
        while(len(self.all_buckets[class_name].get(self.buckets[density_idx],[])) <= 1):
            density_idx -= 1
            if density_idx == -1:
                density_idx = 0 
                while(len(self.all_buckets[class_name].get(self.buckets[density_idx],[])) <= 1):
                    density_idx += 1
                    if density_idx >= len(self.buckets):
                        raise Exception("Infinite loop will occur in get_random_frame_even()")
        
        return self.all_buckets[class_name][self.buckets[density_idx]],density_idx

    def special_log(self,n):
        if n == 0:
            return -1
        return np.log2(n)

    def load_points(self, info, frame_idx):
        points = []
        if 'pts_data' in info:
            for name in self.load_feats:
                points.append(info['pts_data'][f'pts_{name}'])
        elif 'path' in info:
            path = info['path']
            try:
                self.obj_id_to_nums[info['id']]
            except KeyError: 
                self.obj_id_to_nums[info['id']] = self.get_filtered_nums(info)
            path = osp.join(info['path'], frame_idx)
            #self.obj_id_to_nums[info['id']][frame_idx])
            for name, dim in zip(self.load_feats, self.load_dims):
                feats_file = f'{self.data_root}/{path}/pts_{name}.bin'
                num_pts = int(os.stat(feats_file).st_size // (4 * dim))
                num_pts -= int(num_pts * self.load_fraction)
                points.append(np.fromfile(feats_file,
                                          offset=4 * dim * num_pts,
                                          dtype=np.float32).reshape(-1, dim))
        else:
            raise ValueError(f'info must have either path or pts_data')
        return np.concatenate(points, axis=-1)

    def load_image(self, info, frame_idx):
        if 'path' in info:
            path = info['path']
            try:
                self.obj_id_to_nums[info['id']]
            except KeyError: 
                self.obj_id_to_nums[info['id']] = self.get_filtered_nums(info)

            # print(os.listdir(self.data_root))
            path = osp.join(info['path'], frame_idx)
            feats_file = f'{self.data_root}/{path}/img_crop.bin'

            try:
                im = np.fromfile(feats_file,dtype=np.float32).reshape((-1,)+info['crop_size'])
                if tuple(im.shape[1:]) != self.crop_size:
                    im = im[:,0:self.crop_size[0],0:self.crop_size[1]]

            except FileNotFoundError:
                # print('[WARNING] FileNotFoundError in ObjectImageLoaderSparseWaymo No such file "{}" loading ZEROs image instead...'.format(
                #     feats_file))
                im = np.zeros((3,) + self.crop_size)
            except KeyError:
                print('[WARNING] KeyError in ObjectImageLoaderSparseWaymo for crop_size loading ZEROs image instead...')
                im = np.zeros((3,) + self.crop_size)

            return im #np.swapaxes(im.T,0,1)
            
    
        else:
            raise ValueError(f'info must have path')
    
############################################
# Jeongok
############################################
    
@DATASETS.register_module()
class ObjectLoaderSparseJeongok(ObjectLoaderSparseBase):
    def __init__(self,
                 train,
                 metadata_path,
                 version,
                 *args,
                 use_metdata_fix=False,
                 use_jeongok_split=False,
                 **kwargs):

        # Load Jeongok metadata with train/val split
        metadata = load_jeongok_metadata(metadata_path, train)
        
        super().__init__(*args, metadata=metadata, **kwargs)

        # The metadata already contains the obj_infos, so we don't need to manually parse
        # Just ensure all objects have the required fields
        for obj_id, obj_info in self.obj_infos.items():
            # Ensure path is set correctly
            if 'path' not in obj_info:
                obj_info['path'] = f'objects/{obj_id}'
            
            # Ensure class_name is set
            if 'class_name' not in obj_info:
                obj_info['class_name'] = 'boat'
            
            # Ensure scene_id is set
            if 'scene_id' not in obj_info:
                obj_info['scene_id'] = 'jeongok_scene'

        self.obj_id_to_nums = self.collect_obj_id_to_nums(self.min_points)

        t1 = time.time()
        self.get_buckets(np.arange(0,len(self.obj_id_to_nums)))
        self.get_all_buckets(np.arange(0,len(self.obj_id_to_nums)))
        # print("[ObjectLoaderSparseJeongok] Loading buckets took: ",time.time()-t1)

    def load(self,*args,**kwerags):
        return self.load_points(*args,**kwerags)
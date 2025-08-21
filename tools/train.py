import argparse
import copy
import os
import random
import time
import datetime

import os.path as osp
import numpy as np
import torch

# from mmcv import Config
from mmengine.config import Config

from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

from tools.utils import setup_tensorboard_logger
from tools.tqdm_progress_hook import TQDMProgressBarHook

def print_dataset_info(datasets, cfg, dataloader_kwargs):
    """Print detailed information about training and validation datasets."""
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    # Training dataset info
    print("\n📊 TRAINING DATASET:")
    print("-" * 50)
    for i, dataset in enumerate(datasets):
        print(f"Dataset {i+1}: {type(dataset).__name__}")
        print(f"  • Total samples: {len(dataset):,}")
        print(f"  • Dataset class: {dataset.__class__.__name__}")
        
        # Print detailed frame information
        if hasattr(dataset, 'sparse_loader') and hasattr(dataset.sparse_loader, 'obj_infos'):
            total_frames = 0
            total_objects = len(dataset.sparse_loader.obj_infos)
            print(f"  • Total objects: {total_objects}")
            print(f"  • Frame counts per object:")
            for obj_id, obj_info in dataset.sparse_loader.obj_infos.items():
                frame_count = len(obj_info['num_pts'])
                total_frames += frame_count
                print(f"    - {obj_id}: {frame_count} frames")
            print(f"  • Total frames across all objects: {total_frames:,}")
        
        # Print dataset attributes if available
        if hasattr(dataset, 'dataset'):
            print(f"  • Base dataset: {type(dataset.dataset).__name__}")
            if hasattr(dataset.dataset, 'ann_file'):
                print(f"  • Annotation file: {dataset.dataset.ann_file}")
        
        # Print data loading configuration
        train_kwargs = dataloader_kwargs.get('train', {})
        print(f"  • Batch size: {cfg.data.samples_per_gpu}")
        print(f"  • Workers: {cfg.data.workers_per_gpu}")
        print(f"  • Prefetch factor: {train_kwargs.get('prefetch_factor', 'N/A')}")
        print(f"  • Shuffle: {train_kwargs.get('shuffle', 'N/A')}")
        print(f"  • Drop last: {train_kwargs.get('drop_last', 'N/A')}")
        print(f"  • Persistent workers: {train_kwargs.get('persistent_workers', 'N/A')}")
    
    # Validation dataset info
    if cfg.validate:
        print("\n📊 VALIDATION DATASET:")
        print("-" * 50)
        try:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            print(f"  • Total samples: {len(val_dataset):,}")
            print(f"  • Dataset class: {val_dataset.__class__.__name__}")
            
            # Print detailed frame information for validation
            if hasattr(val_dataset, 'sparse_loader') and hasattr(val_dataset.sparse_loader, 'obj_infos'):
                total_frames = 0
                total_objects = len(val_dataset.sparse_loader.obj_infos)
                print(f"  • Total objects: {total_objects}")
                print(f"  • Frame counts per object:")
                for obj_id, obj_info in val_dataset.sparse_loader.obj_infos.items():
                    frame_count = len(obj_info['num_pts'])
                    total_frames += frame_count
                    print(f"    - {obj_id}: {frame_count} frames")
                print(f"  • Total frames across all objects: {total_frames:,}")
            
            if hasattr(val_dataset, 'dataset'):
                print(f"  • Base dataset: {type(val_dataset.dataset).__name__}")
                if hasattr(val_dataset.dataset, 'ann_file'):
                    print(f"  • Annotation file: {val_dataset.dataset.ann_file}")
            
            # Print validation data loading configuration
            val_kwargs = dataloader_kwargs.get('val', {})
            val_samples_per_gpu = cfg.data.get("val_samples_per_gpu", 1)
            print(f"  • Batch size: {val_samples_per_gpu}")
            print(f"  • Workers: {cfg.data.workers_per_gpu}")
            print(f"  • Prefetch factor: {val_kwargs.get('prefetch_factor', 'N/A')}")
            print(f"  • Shuffle: {val_kwargs.get('shuffle', 'N/A')}")
            print(f"  • Persistent workers: {val_kwargs.get('persistent_workers', 'N/A')}")

            # Print number of positive and negative pairs for validation dataset
            if hasattr(val_dataset, 'val_positives') and hasattr(val_dataset, 'val_negatives'):
                print(f"  • Number of positive pairs: {len(val_dataset.val_positives)}")
                print(f"  • Number of negative pairs: {len(val_dataset.val_negatives)}")
                # Print which objects are used for positive and negative pairs
                pos_obj_set = set([sample['tok'] for sample in val_dataset.val_positives])
                neg_obj_set = set([sample['tok2'] for sample in val_dataset.val_negatives])
                print(f"  • Objects used for positive pairs: {sorted(pos_obj_set)}")
                print(f"  • Objects used as negative pair targets: {sorted(neg_obj_set)}")
                # Export all validation pairs to a timestamped txt file
                export_val_pairs(val_dataset, output_dir=cfg.run_dir)
            
        except Exception as e:
            print(f"  • Error building validation dataset: {e}")
    
    # Model configuration
    print("\n🤖 MODEL CONFIGURATION:")
    print("-" * 50)
    print(f"  • Eval only: {cfg.model.get('eval_only', 'N/A')}")
    print(f"  • Triplet sample num: {cfg.model.get('triplet_sample_num', 'N/A')}")
    
    # Evaluation configuration
    print("\n📈 EVALUATION CONFIGURATION:")
    print("-" * 50)
    eval_cfg = cfg.get("evaluation", {})
    print(f"  • Interval: {eval_cfg.get('interval', 'N/A')}")
    print(f"  • Start epoch: {eval_cfg.get('start', 'N/A')}")
    print(f"  • Pipeline: {eval_cfg.get('pipeline', 'N/A')}")
    
    # Training configuration
    print("\n⚙️ TRAINING CONFIGURATION:")
    print("-" * 50)
    print(f"  • Total epochs: {cfg.runner.get('max_epochs', 'N/A')}")
    print(f"  • Learning rate: {cfg.optimizer.get('lr', 'N/A')}")
    print(f"  • Optimizer: {cfg.optimizer.get('type', 'N/A')}")
    print(f"  • Seed: {cfg.seed}")
    print(f"  • Deterministic: {cfg.deterministic}")
    
    print("\n" + "="*80)
    print()

def export_val_pairs(val_dataset, output_dir="."):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"val_pairs_{now}.txt")
    with open(output_path, "w") as f:
        # Positive pairs
        if hasattr(val_dataset, 'val_positives'):
            for sample in val_dataset.val_positives:
                obj = sample['tok']
                frame1 = sample['o1']
                frame2 = sample['o2']
                cls = sample['cls']
                npts1 = val_dataset.sparse_loader.obj_infos[obj]['num_pts'][frame1]
                npts2 = val_dataset.sparse_loader.obj_infos[obj]['num_pts'][frame2]
                f.write(f"POS\t{obj}\t{frame1}\t{npts1}\t{frame2}\t{npts2}\tclass:{cls}\n")
        # Negative pairs
        if hasattr(val_dataset, 'val_negatives'):
            for sample in val_dataset.val_negatives:
                obj1 = sample['tok1']
                frame1 = sample['o1']
                obj2 = sample['tok2']
                frame2 = sample['o2']
                cls1 = sample['cls1']
                cls2 = sample['cls2']
                npts1 = val_dataset.sparse_loader.obj_infos[obj1]['num_pts'][frame1]
                npts2 = val_dataset.sparse_loader.obj_infos[obj2]['num_pts'][frame2]
                f.write(f"NEG\t{obj1}\t{frame1}\t{npts1}\t{obj2}\t{frame2}\t{npts2}\tclass1:{cls1}\tclass2:{cls2}\n")
    print(f"Validation pairs exported to {output_path}")

def main():

    # Suppress NCCL verbose output
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'WARN'
    
    # Suppress MMCV debug output
    os.environ['MMCV_LOG_LEVEL'] = 'WARNING'
    os.environ['MMCV_DISABLE_HOOK_DEBUG'] = '1'

    # Suppress PyTorch deprecation warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*size_average.*")
    warnings.filterwarnings("ignore", message=".*reduce.*")

    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help='the checkpoint file for evaluation',
    )
    parser.add_argument(
        "--tensorboard-prefix",
        type=str,
        default=None,
        help='prefix for tensorboard logging',
    )
    args, opts = parser.parse_known_args()

    if 'configs_reid' in args.config.split('/')[0]:
        print("Loading config from configs_reid: ", args.config)
        cfg = Config.fromfile(args.config)
        cfg = setup_tensorboard_logger(cfg, args, args.tensorboard_prefix, args.checkpoint)
        dataloader_kwargs=cfg.dataloader_kwargs
    else:
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg = Config(recursive_eval(configs), filename=args.config)
        dataloader_kwargs=dict(shuffle=True, prefetch_factor=4)
    
    if args.checkpoint is not None:
        print("loading from checkpoint:", args.checkpoint)
        cfg.load_from = args.checkpoint

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    print("\n\n")
    print("###############################################################################################")
    print("Setting local rank (GPU ID) to {}".format(dist.local_rank()))
    print("###############################################################################################")
    print("\n\n")

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    # logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        # logger.info(
        #     f"Set random seed to {cfg.seed}, "
        #     f"deterministic mode: {cfg.deterministic}"
        # )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # print("CFG_DATA_TRAIN: ", cfg.data.train)
    datasets = [build_dataset(cfg.data.train)]

    # Print dataset information
    print_dataset_info(datasets, cfg, dataloader_kwargs)

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    # logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=cfg.validate,
        timestamp=timestamp,
        dataloader_kwargs=dataloader_kwargs,
    )


if __name__ == "__main__":
    main()

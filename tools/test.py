import argparse
import copy
import os
import warnings

import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval

from tools.utils import setup_tensorboard_logger


def print_test_dataset_info(dataset, cfg, data_loader, args):
    """Print detailed information about test dataset."""
    print("\n" + "="*80)
    print("TEST DATASET INFORMATION")
    print("="*80)
    
    print("\n📊 TEST DATASET:")
    print("-" * 50)
    print(f"  • Total samples: {len(dataset):,}")
    print(f"  • Dataset class: {dataset.__class__.__name__}")
    
    # Print dataset attributes if available
    if hasattr(dataset, 'dataset'):
        print(f"  • Base dataset: {type(dataset.dataset).__name__}")
        if hasattr(dataset.dataset, 'ann_file'):
            print(f"  • Annotation file: {dataset.dataset.ann_file}")
    
    # Print data loading configuration
    print(f"  • Batch size: {cfg.data.val_samples_per_gpu}")
    print(f"  • Workers: {cfg.data.workers_per_gpu}")
    print(f"  • Shuffle: False (testing)")
    print(f"  • Distributed: True")
    
    # Print dataloader info
    print(f"  • Dataloader length: {len(data_loader)}")
    print(f"  • Samples per GPU: {cfg.data.val_samples_per_gpu}")
    
    # Model configuration
    print("\n🤖 MODEL CONFIGURATION:")
    print("-" * 50)
    print(f"  • Eval only: {cfg.model.get('eval_only', 'N/A')}")
    print(f"  • Triplet sample num: {cfg.model.get('triplet_sample_num', 'N/A')}")
    
    # Test configuration
    print("\n🧪 TEST CONFIGURATION:")
    print("-" * 50)
    print(f"  • Checkpoint: {args.checkpoint}")
    print(f"  • Fuse conv-bn: {args.fuse_conv_bn}")
    print(f"  • Format only: {args.format_only}")
    print(f"  • Show results: {args.show}")
    print(f"  • Show dir: {args.show_dir}")
    print(f"  • Evaluation metrics: {args.eval}")
    
    # Evaluation configuration
    print("\n📈 EVALUATION CONFIGURATION:")
    print("-" * 50)
    eval_cfg = cfg.get("evaluation", {})
    print(f"  • Interval: {eval_cfg.get('interval', 'N/A')}")
    print(f"  • Start epoch: {eval_cfg.get('start', 'N/A')}")
    print(f"  • Pipeline: {eval_cfg.get('pipeline', 'N/A')}")
    
    print("\n" + "="*80)
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():

    # Suppress PyTorch deprecation warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*size_average.*")
    warnings.filterwarnings("ignore", message=".*reduce.*")

    # assert '/btherien/github/nuscenes-devkit/python-sdk' in os.environ['PYTHONPATH']
    args = parse_args()

    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    print(args.config.split('/')[0])

    if 'tracking' in args.config.split('/')[0]:
        cfg = Config.fromfile(args.config)
        cfg = setup_tensorboard_logger(cfg,args)
        dataloader_kwargs=cfg.dataloader_kwargs
    else:
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg = Config(recursive_eval(configs), filename=args.config)
        dataloader_kwargs=dict(shuffle=True, prefetch_factor=4)


    
    # print(cfg.pretty_text)


    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.val, dict):
        cfg.data.val.test_mode = True
        samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
    elif isinstance(cfg.data.val, list):
        for ds_cfg in cfg.data.val:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.val]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.val:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = True
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    else:
        set_random_seed(cfg.seed, deterministic=args.deterministic)



    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)


    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # Print dataset information
    print_test_dataset_info(dataset, cfg, data_loader, args)

    # old versions did not save class info in checkpoints, this walkaround is
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()

# Define the class mappings (required)
tracking_classes = {'boat':'boat','FP_boat':'FP_boat'}
tracking_classes_fp = {'boat':'boat','FP_boat':'FP_boat'}
cls_to_idx = {'none_key': -1, 'boat': 0}
cls_to_idx_fp = {'none_key': -1, 'boat': 0, 'FP_boat': 1}
CLASSES = ['boat']

resume_from = None

data = dict(
    samples_per_gpu = 8,
    val_samples_per_gpu = 16, # (val_)samples_per_gpu < subsample_sparse <= min_points
    workers_per_gpu = 4,
    train=dict(
        type='ReIDDatasetJeongok',
        cls_to_idx=cls_to_idx,
        cls_to_idx_fp=cls_to_idx_fp,
        tracking_classes=tracking_classes,
        tracking_classes_fp=tracking_classes_fp,
        CLASSES=CLASSES,
        subsample_sparse=256,
        subsample_mode="random",
        val_subsample_mode="random",
        return_mode='dict',
        verbose=False,
        validation_seed=0,
        pairs_per_target=20,  # Number of positive pairs per class per epoch (with matched negatives)
        augmentations=["rotate", "translate", "jitter"],  # List of augmentations to use
        sparse_loader=dict(
            type='ObjectLoaderSparseJeongok',
            train=True,
            tracking_classes=tracking_classes,
            metadata_path='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/metadata/metadata.pkl',
            version='v1.0-trainval',
            data_root='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/objects',
            min_points=2,
            load_feats=['xyz', 'intensity'],
            load_dims=[4],
            use_jeongok_split=True
        )
    ),
    val=dict(
        type='ReIDDatasetJeongokValEven',
        cls_to_idx=cls_to_idx,
        cls_to_idx_fp=cls_to_idx_fp,
        tracking_classes=tracking_classes,
        tracking_classes_fp=tracking_classes_fp,
        CLASSES=CLASSES,
        subsample_sparse=256,
        subsample_mode="random",
        val_subsample_mode="random",
        return_mode='dict',
        verbose=False,
        validation_seed=0,
        max_combinations=20,
        sparse_loader=dict(
            type='ObjectLoaderSparseJeongok',
            train=False,
            tracking_classes=tracking_classes,
            metadata_path='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/metadata/metadata.pkl',
            version='v1.0-trainval',
            data_root='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/objects',
            min_points=2,
            load_feats=['xyz', 'intensity'],
            load_dims=[4],
            use_jeongok_split=True
        )
    )
)
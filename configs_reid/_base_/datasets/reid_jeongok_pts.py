# Define the class mappings (required)
tracking_classes = {'boat': 'boat'}
tracking_classes_fp = {'boat': 'boat'}
cls_to_idx = {'none_key': -1, 'boat': 0}
cls_to_idx_fp = {'none_key': -1, 'boat': 0, 'FP_boat': 1}
CLASSES = ['boat']

# Feature selection for point cloud input
point_feature_type = 'xyz'  # 'xyz' or 'xyz_eigen'
point_feature_dim = 3 if point_feature_type == 'xyz' else 6
eigen_k = 16  # Only used if point_feature_type == 'xyz_eigen'

data = dict(
    train=dict(
        type='ReIDDatasetJeongok',
        cls_to_idx=cls_to_idx,
        cls_to_idx_fp=cls_to_idx_fp,
        tracking_classes=tracking_classes,
        tracking_classes_fp=tracking_classes_fp,
        CLASSES=CLASSES,
        subsample_sparse=128,
        subsample_mode="random",
        val_subsample_mode="fps",
        return_mode='dict',
        verbose=False,
        validation_seed=0,
        num_pairs_per_object=50,  # Number of pairs per object per epoch
        augmentations=None,
        # augmentations=["rotate", "translate", "jitter"],  # List of augmentations to use
        sparse_loader=dict(
            type='ObjectLoaderSparseJeongok',
            train=True,
            tracking_classes=tracking_classes,
            metadata_path='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both/metadata/metadata.pkl',
            version='v1.0-trainval',
            data_root='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both',
            min_points=2,
            load_feats=[point_feature_type],
            load_dims=[point_feature_dim],
            eigen_k=eigen_k,  # Pass k for eigen if needed
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
        subsample_sparse=128,
        subsample_mode="random",
        val_subsample_mode="fps",
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
            data_root='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both',
            min_points=2,
            load_feats=['xyz'],
            load_dims=[3],
            use_jeongok_split=True
        )
    )
)

# tracking_classes = {
#     'boat':'boat'
# }

# tracking_classes_fp = {
#     'boat':'boat'
# }

# cls_to_idx = {
#     'none_key':-1,
#     'boat':0
# }

# cls_to_idx_fp = {
#     'none_key':-1,
#     'boat':0,
#     'FP_boat':1
# }

# CLASSES = ['boat']

# version = 'trainval'
# train_metadata_version = 'trainval-det-both'
# val_metadata_version = 'trainval-det-both'

# data = dict(
#     samples_per_gpu=256,
#     workers_per_gpu=8,
#     train=dict(type='ReIDDatasetJeongok',
#                cls_to_idx=cls_to_idx,
#                cls_to_idx_fp=cls_to_idx_fp,
#                tracking_classes=tracking_classes,
#                tracking_classes_fp=tracking_classes_fp,
#                subsample_sparse=128,
#                subsample_mode="random",
#                val_subsample_mode="fps",
#                CLASSES=CLASSES,
#                return_mode='dict',
#                verbose=False,
#                validation_seed=0,
#                sparse_loader=dict(type='ObjectLoaderSparseJeongok',
#                                 train=True,
#                                 version='v1.0-{}'.format(version),
#                                 tracking_classes=tracking_classes,
#                                 metadata_path='Datasets/Jeongok-ReID/data/lstk/sparse-{}/metadata/metadata.pkl'.format(train_metadata_version),
#                                 data_root='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both',
#                                 min_points=2,
#                                 load_scene=True,
#                                 load_objects=True,
#                                 load_feats=['xyz'],
#                                 load_dims=[3],
#                                 use_jeongok_split=True)  # Add flag to use Jeongok-specific split
#             ),
#     val=dict(type='ReIDDatasetJeongokValEven',
#                cls_to_idx=cls_to_idx,
#                cls_to_idx_fp=cls_to_idx_fp,
#                tracking_classes=tracking_classes,
#                tracking_classes_fp=tracking_classes_fp,
#                subsample_sparse=128,
#                subsample_mode="random",
#                val_subsample_mode="fps",
#                CLASSES=CLASSES,
#                return_mode='dict',
#                verbose=False,
#                validation_seed=0,
#                max_combinations=10,
#                sparse_loader=dict(type='ObjectLoaderSparseJeongok',
#                                 train=False,
#                                 version='v1.0-{}'.format(version),
#                                 tracking_classes=tracking_classes,
#                                 metadata_path='Datasets/Jeongok-ReID/data/lstk/sparse-{}/metadata/metadata.pkl'.format(val_metadata_version),
#                                 data_root='Datasets/Jeongok-ReID/data/lstk/sparse-trainval-det-both',
#                                 min_points=2,
#                                 load_scene=True,
#                                 load_objects=True,
#                                 load_feats=['xyz'],
#                                 load_dims=[3],
#                                 use_jeongok_split=True))  # Add flag to use Jeongok-specific split
# )
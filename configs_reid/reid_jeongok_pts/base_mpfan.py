_base_ = [
    "../_base_/datasets/reid_jeongok_pts.py",
    "../_base_/reidentifiers/reid_pts_mpfan.py",
]

model = dict(
    type='ReIDNet',

    # backbone=dict(type='PTr-PN', fe_module='pointnet'),
    # backbone=dict(type='PTr-PX', fe_module='pointnext'),
    # backbone=dict(type='PTr-DG', fe_module='dgcnn'),
    # backbone=dict(type='PTr-Deep', fe_module='deepgcn'),
    backbone=dict(type='PTr-DG-ED', fe_module='dgcnn', ED_nsample=10, ED_conv_out=16),
    # backbone=dict(type='PTr-Deep-ED', fe_module='deepgcn', ED_nsample=10, ED_conv_out=16),

    losses_to_use=dict(
        match=True,
        kl=False,
        triplet=False,
        fp=False,  #BCE
        cls=False, #CE
        ),
)

_bs = 128
_min_points = 128
_subsample_mode = "random" # random | fps | rand_crop
_val_subsample_mode = "fps" # random | fps | rand_crop
_subsample_sparse = 128

data = dict(
    samples_per_gpu = _bs,
    val_samples_per_gpu = _bs*2, # (val_)samples_per_gpu < subsample_sparse <= min_points
    workers_per_gpu = 4,
    train = dict(
        subsample_sparse = _subsample_sparse,
        subsample_mode = _subsample_mode,
        val_subsample_mode=_val_subsample_mode,
        sparse_loader = dict(
            min_points = _min_points
        ),
    ),
    val = dict(
        subsample_sparse = _subsample_sparse,
        subsample_mode = _subsample_mode,
        val_subsample_mode=_val_subsample_mode,
        sparse_loader = dict(
            min_points = _min_points
        ),
    ),
)

resume_from = None
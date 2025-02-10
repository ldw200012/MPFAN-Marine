_base_ = [
    "../base_pointnext.py",
    "../../_base_/schedules/cyclic_500e_lr3e-4_norm1.py",
    "../../_base_/reidentification_runtime.py",
]

model = dict(
    eval_only=False,
    triplet_sample_num=128,
)

evaluation = dict(interval=10, pipeline=[], start=0)

dataloader_kwargs = dict(
    val=dict(shuffle=True, prefetch_factor=2,persistent_workers=True),
    train=dict(shuffle=True, prefetch_factor=8,persistent_workers=True,drop_last=True))

neptune_tags = ['nus','500e','4 x 256','PointNeXt','only match','point-cat']
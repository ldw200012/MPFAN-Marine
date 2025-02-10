_base_ = [
    "../base_pointtransformer.py",
    "../../_base_/schedules/cyclic_500e_lr1e-5.py",
    "../../_base_/reidentification_runtime_testing.py",
]

model = dict(
    eval_only=True,
    triplet_sample_num=32,
)

evaluation = dict(interval=1, pipeline=[], start=0)

dataloader_kwargs = dict(
    val=dict(shuffle=True, prefetch_factor=36,persistent_workers=True),
    train=dict(shuffle=True, prefetch_factor=18,persistent_workers=True))

neptune_tags = ['nus','only-match','testing','PointTransformer','point-cat']
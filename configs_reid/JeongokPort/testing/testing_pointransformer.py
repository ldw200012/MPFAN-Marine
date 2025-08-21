_base_ = [
    "../../_base_/datasets/reid_jeongok_pts.py",
    "../../_base_/schedules/cyclic_500e_lr1e-5.py",
    "../../_base_/reidentification_runtime_testing.py",
    "../../_base_/reidentifiers/reid_pts_pointtransformer.py",
]

model = dict(
    eval_only=True,
    triplet_sample_num=32,
)
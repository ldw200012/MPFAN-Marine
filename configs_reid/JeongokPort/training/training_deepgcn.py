_base_ = [
    "../../_base_/datasets/reid_jeongok_pts.py",
    "../../_base_/schedules/cyclic_500e_lr3e-4_norm1.py",
    "../../_base_/reidentification_runtime.py",
    "../../_base_/reidentifiers/reid_pts_deepgcn.py",
]

model = dict(
    eval_only=False,
    triplet_sample_num=128,
)
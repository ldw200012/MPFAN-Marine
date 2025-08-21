_base_ = ['./default_runtime.py']

checkpoint_config = dict(interval=5,max_keep_ckpts=1,save_last=True,save_optimizer=True)

evaluation = dict(interval=1, pipeline=[], start=0)

dataloader_kwargs = dict(
    val=dict(shuffle=True, prefetch_factor=36,persistent_workers=True),
    train=dict(shuffle=True, prefetch_factor=18,persistent_workers=True))
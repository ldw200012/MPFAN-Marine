_base_ = ['./default_runtime.py']

checkpoint_config = dict(interval=500,max_keep_ckpts=-1,save_last=True,save_optimizer=True)

evaluation = dict(interval=10, save_best='auto', rule='greater', pipeline=[], start=0)

dataloader_kwargs = dict(
    val=dict(shuffle=True, prefetch_factor=2,persistent_workers=True),
    train=dict(shuffle=True, prefetch_factor=8,persistent_workers=True,drop_last=True))
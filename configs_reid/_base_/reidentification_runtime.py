_base_ = ['./default_runtime.py']


workflow = [('train', 1)]#,('val',1)]
work_dir='work_dirs'

checkpoint_config = dict(interval=5,max_keep_ckpts=1,save_last=True,save_optimizer=True)
evaluation = dict(interval=10, pipeline=[], start=0)
find_unused_parameters=True
dataloader_shuffle=False


cudnn_benchmark=False
dataloader_kwargs = dict(shuffle=True, prefetch_factor=6,persistent_workers=True)
train_tracker = False
seed=66
deterministic=False

log_config = dict(interval=8,
                hooks=[
                    dict(type='TextLoggerHook',reset_flag=False),
                    dict(type="NeptuneLoggerHook",
                        init_kwargs={
                            'project':"dongwooklee1201/mpfan-marine-eval",
                            'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNmY3MjBiMi01Mjg2LTQwOTYtODIzYy00Mjk4MGIwMTQ4ZjcifQ==",
                            'name': "",
                            'source_files': [],
                            'tags': []
                            },
                        interval=16,
                        ignore_last=True,
                        reset_flag=True,
                        with_step=True,
                        by_epoch=True),
            ])

validate=True

custom_hooks = [
    dict(type='SaveModelToNeptuneHook', priority=40),
]
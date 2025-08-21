checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook

log_config = dict(
    interval=8,
    hooks=[
        dict(type='TextLoggerHook',reset_flag=False),
        dict(type="TensorboardLoggerHook",
            log_dir='runs/tensorboard',
            interval=16),
])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]#,('val',1)]
work_dir='work_dirs'
find_unused_parameters=True
dataloader_shuffle=False
cudnn_benchmark=False
train_tracker = False
seed=66
deterministic=False
validate=True

custom_hooks = [
    dict(type='SaveModelToTensorboardHook', priority=40),
    dict(type='TQDMProgressBarHook')
]
import os.path as osp
import time


def setup_tensorboard_logger(cfg, args, tensorboard_prefix=None, checkpoint=None):
    try:
        cfg.dataloader_kwargs
    except AttributeError:
        raise AttributeError("You need to add 'dataloader_kwargs' to your config file. See 'configs/_base_/reidentification_runtine.py' for an example.")

    try:
        cfg.train_tracker
    except AttributeError:
        raise AttributeError("You need to add 'train_tracker' to your config file. See 'configs/_base_/reidentification_runtine.py' for an example.")
        

    if cfg.train_tracker:
        assert cfg.dataloader_kwargs['shuffle'] == False, "You need to set 'dataloader_kwargs.shuffle' to False when training a tracker."

    # Check if TensorBoard logger is configured
    tensorboard_hook_found = False
    for hook in cfg.log_config.hooks:
        if hook.get('type') == 'TensorboardLoggerHook':
            tensorboard_hook_found = True
            break
    
    if not tensorboard_hook_found:
        print('###############################################################################################')
        print('\t WARNING : No TensorboardLoggerHook in config file. This run will not be logged to TensorBoard.')
        print('###############################################################################################')
        time.sleep(3)

    return cfg
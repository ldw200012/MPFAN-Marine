from mmcv.runner import HOOKS, Hook
from tqdm import tqdm

@HOOKS.register_module()
class TQDMProgressBarHook(Hook):
    def before_train_epoch(self, runner):
        # dataset_len = len(runner.data_loader.dataset)
        # batch_size = runner.data_loader.batch_size
        # print(f"[TQDMProgressBarHook] Dataset length: {dataset_len}, Batch size: {batch_size}")
        self.pbar = tqdm(total=len(runner.data_loader), desc=f"Epoch {runner.epoch+1}", unit='samples')

    def after_train_iter(self, runner):
        batch_size = runner.data_loader.batch_size
        self.pbar.update(batch_size)

    def after_train_epoch(self, runner):
        self.pbar.close() 
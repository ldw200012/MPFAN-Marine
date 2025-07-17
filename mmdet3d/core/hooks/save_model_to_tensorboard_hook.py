import torch
import torch.distributed as dist
import os
from mmcv.runner import HOOKS, Hook, EvalHook, get_dist_info
from torch.utils.tensorboard import SummaryWriter


@HOOKS.register_module()
class SaveModelToTensorboardHook(Hook):

    def __init__(self, log_dir='runs/tensorboard'):
        self.log_dir = log_dir
        self.writer = None

    def before_run(self, runner):
        """Initialize TensorBoard writer."""
        rank, world_size = get_dist_info()
        if world_size == 1 or runner.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)

    def after_run(self, runner):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def save_model_tensorboard(self, runner):
        rank, world_size = get_dist_info()
        if world_size == 1:
            self.save_model_tensorboard_(runner)
        else:
            if runner.rank == 0:
                self.save_model_tensorboard_(runner)
            dist.barrier()

    def save_model_tensorboard_(self, runner):
        """Save model checkpoint and log to TensorBoard."""
        print('Saving model in SaveModelToTensorboardHook')
        
        # Save checkpoint to work_dir
        checkpoint_path = os.path.join(runner.work_dir, f'epoch_{runner.epoch + 1}.pth')
        runner.save_checkpoint(out_dir=runner.work_dir,
                             filename_tmpl='epoch_{}.pth',
                             save_optimizer=True,
                             meta=None,
                             create_symlink=False)
        
        print(f"[Info] Model saved to {checkpoint_path}")
        
        # Log model file size to TensorBoard
        if self.writer is not None and os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Convert to MB
            self.writer.add_scalar('Model/Checkpoint_Size_MB', file_size, runner.epoch)
            print(f"[Info] Model size logged to TensorBoard: {file_size:.2f} MB")


@HOOKS.register_module()
class LogConfigToTensorboardHook(Hook):

    def __init__(self, log_dir='runs/tensorboard'):
        self.log_dir = log_dir
        self.writer = None

    def before_run(self, runner):
        """Initialize TensorBoard writer and log configuration."""
        rank, world_size = get_dist_info()
        if world_size == 1 or runner.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            self.log_config_(runner)

    def after_run(self, runner):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()

    def log_config_(self, runner):
        """Log configuration parameters to TensorBoard."""
        if self.writer is not None:
            # Log training parameters
            self.writer.add_scalar('Config/Max_Iterations', runner._max_iters, 0)
            self.writer.add_scalar('Config/Max_Epochs', runner._max_epochs, 0)
            
            # Log model configuration if available
            if hasattr(runner.model, 'module'):
                model = runner.model.module
                if hasattr(model, 'trackManager'):
                    self.writer.add_text('Config/Model_Type', str(type(model)), 0)
            
            print("[Info] Configuration logged to TensorBoard") 
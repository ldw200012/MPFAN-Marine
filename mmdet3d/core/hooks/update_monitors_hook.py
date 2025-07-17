from mmcv.runner import HOOKS, Hook, get_dist_info
import torch.distributed as dist


@HOOKS.register_module()
class UpdateGradMonitor(Hook):

    def __init__(self):
        pass

    def after_train_iter(self, runner):
        self.update_grad_monitor(runner)

    def update_grad_monitor(self,runner):
        rank, world_size = get_dist_info()
        if world_size == 1:
            self.update_grad_monitor_(runner)
        else:
            if runner.rank == 0:
                self.update_grad_monitor_(runner)
            dist.barrier()

    def update_grad_monitor_(self,runner):
        # Find TensorBoard writer from hooks
        tensorboard_writer = None
        for hook in runner._hooks:
            if hasattr(hook, 'writer') and hook.writer is not None:
                tensorboard_writer = hook.writer
                break

        runner.model.module.trackManager.update_grad_monitor(tensorboard_writer)


@HOOKS.register_module()
class UpdateParamMonitor(Hook):

    def __init__(self):
        pass

    def after_train_iter(self, runner):
        self.update_param_monitor(runner)

    def update_param_monitor(self,runner):
        rank, world_size = get_dist_info()
        if world_size == 1:
            self.update_param_monitor_(runner)
        else:
            if runner.rank == 0:
                self.update_param_monitor_(runner)
            dist.barrier()

    def update_param_monitor_(self,runner):
        # Find TensorBoard writer from hooks
        tensorboard_writer = None
        for hook in runner._hooks:
            if hasattr(hook, 'writer') and hook.writer is not None:
                tensorboard_writer = hook.writer
                break

        runner.model.module.trackManager.update_param_monitor(tensorboard_writer)
        






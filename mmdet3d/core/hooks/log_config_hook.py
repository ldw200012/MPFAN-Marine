from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class UploadConfig(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        self.upload_config(runner)

    def upload_config(self, runner):
        rank, world_size = get_dist_info()
        if world_size == 1:
            self.upload_config_(runner)
        else:
            if runner.rank == 0:
                self.upload_config_(runner)
            dist.barrier()

    def upload_config_(self, runner):
        # Find TensorBoard writer from hooks
        tensorboard_writer = None
        for hook in runner._hooks:
            if hasattr(hook, 'writer') and hook.writer is not None:
                tensorboard_writer = hook.writer
                break

        if tensorboard_writer is not None and hasattr(runner, 'cfg_path'):
            # Log config file content to TensorBoard
            try:
                with open(runner.cfg_path, 'r') as f:
                    config_content = f.read()
                tensorboard_writer.add_text('Config/Config_File', config_content, 0)
                print("[Info] Config file uploaded to TensorBoard")
            except Exception as e:
                print(f"[Warning] Failed to upload config to TensorBoard: {e}")


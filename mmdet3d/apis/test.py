import mmcv
import torch
from tqdm import tqdm


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = tqdm(total=len(dataset), desc='Testing', position=1, leave=False,
                   dynamic_ncols=True, mininterval=1.0, maxinterval=5.0)
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        batch_size = len(result)
        prog_bar.update(batch_size)
    
    prog_bar.close()
    return results

import time
import torch
from torch import inference_mode
from torchvision.ops import complete_box_iou_loss

import time
import torch
from torch import inference_mode
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@inference_mode()
def evaluate(model, dataloader, device):
    '''evaluate model on given data in dataloader
    Args:
        model
        dataloader
        device
    
    Returns:
    '''
    model.eval()
    for images, targets in dataloader:
        images = list(img.to(device) for img in images) # TODO send all at once to gpu
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # start timer and inference
        model_time = time.time()
        predictions = model(images)
        model_time = time.time() - model_time

        metric = MeanAveragePrecision(iou_type="bbox")
        # TODO torchmetrics doesn't support tv_tensors
        for i in range(len(targets)):
            targets[i]['boxes']= torch.Tensor(targets[i]['boxes'])
            
        metric.update(predictions, targets)

        result = metric.compute()
        return result
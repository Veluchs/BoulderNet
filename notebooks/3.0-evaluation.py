# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%

# %%
import time
import torch
from torch import inference_mode
from torchvision.ops import complete_box_iou_loss

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
        images = list(img.to('device') for img in images) # TODO send all at once to gpu
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # start timer and inference
        model_time = time.time()
        predicitions = model(images)
        model_time = time.time() - model_time

        loss = complete_box_iou_loss(predicitions['boxes'], targets['boxes'])
        return loss

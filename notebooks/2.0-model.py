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
#     display_name: bouldernet
#     language: python
#     name: python3
# ---

# %%
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 960

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2,
    )

    # put the pieces together inside a Faster-RCNN model

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                                output_size=14,
                                                                sampling_ratio=2)

    model = MaskRCNN(backbone,
                            num_classes=num_classes,
                            rpn_anchor_generator=anchor_generator,
                            box_roi_pool=roi_pooler,
                            mask_roi_pool=mask_roi_pooler)

    return model


# %%
from torchvision import tv_tensors
import sys

sys.path.insert(1, '../src/')
from torchvision import tv_tensors
from dataset import ClimbingHoldDataset
import torch
from torchvision.models import MobileNet_V3_Large_Weights

# %%
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToDtype(torch.uint8, scale=False),
    v2.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.4, hue=0.3),
    v2.RandomRotation(180),
    v2.ToDtype(torch.float),
    v2.Normalize([0], [255]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# %%

dataset = ClimbingHoldDataset('../data/processed/', transforms=None)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=1,
)



# %%
img = dataset.__getitem__(9)[0]
target = dataset.__getitem__(9)[1]

img.dtype

# %%
import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

import utils

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        

        # reduce losses over all GPUs for logging purposes

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(i)
        if lr_scheduler is not None:
            lr_scheduler.step()
        epoch_loss += losses
    epoch_loss /= i+1
    return epoch_loss

# %%
import torch

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')
import utils
from dataset import ClimbingHoldDataset

from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToDtype(torch.uint8, scale=False),
    v2.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.4, hue=0.3),
    v2.RandomRotation(180),
    v2.ToDtype(torch.float),
    v2.Normalize([0], [255]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(3)
model.to(device)
# use our dataset and defined transformations
dataset = ClimbingHoldDataset('../data/processed/', transforms)

indices = range(len(dataset))
dataset = torch.utils.data.Subset(dataset, indices[:])


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=utils.collate_fn
)


# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 3

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset

# print("That's it!")

# %%
model.eval()
img = dataset.__getitem__(34)[0]
target = dataset.__getitem__(34)[1]
img.shape


# %%
pred = model([img])

# %%
img

# %%
pred

# %%
from utils import show

show([img.to(torch.uint8), pred[0]])

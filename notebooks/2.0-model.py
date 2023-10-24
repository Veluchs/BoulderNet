# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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


# +
from torchvision import tv_tensors
import sys

sys.path.insert(1, '../src/')
from torchvision import tv_tensors
from dataset import ClimbingHoldDataset
import torch
from torchvision.models import MobileNet_V3_Large_Weights

# +
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToDtype(torch.uint8, scale=False),
    v2.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.4, hue=0.3),
    v2.RandomRotation(180),
    v2.ToDtype(torch.float),
    v2.Normalize([0], [255]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# +

dataset = ClimbingHoldDataset('../data/processed/', transforms=None)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=1,
)



# +
img = dataset.__getitem__(9)[0]
target = dataset.__getitem__(9)[1]

img.dtype

# +
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
        # print(i)
        if lr_scheduler is not None:
            lr_scheduler.step()
        epoch_loss += losses
    epoch_loss /= i+1
    return epoch_loss

# +
import torch

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/')
import utils
from dataset import ClimbingHoldDataset

from torchvision.transforms import v2


def get_transform(train):
    transforms = []
    transforms.append(v2.ToDtype(torch.uint8, scale=False)
    transforms.append(v2.ToDtype(torch.float32, scale=True)
                      
    if train:
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
        transforms.append(v2.RandomVerticalFlip(p=0.5))
        transforms.appen(v2.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.4, hue=0.3))
                      
    transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return v2.Compose(transforms)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(3)
model.to(device)
# use our dataset and defined transformations
dataset = ClimbingHoldDataset('/datasets/holds/', get_transforms(train=True)
dataset_test = ClimbingHoldDataset('/datasets/holds/', get_transforms(train=False)



indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=1,
    collate_fn=utils.collate_fn
)

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test,
#     batch_size=1,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=utils.collate_fn


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
num_epochs = 100

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    print(epoch)
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset

# print("That's it!")

# +
from torchvision.transforms import v2
sys.path.insert(1, '../src/')
import utils




inf_transforms = v2.Compose([
    v2.ToDtype(torch.uint8, scale=False),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




dataset = ClimbingHoldDataset('/datasets/holds/', inf_transforms)


img, target = dataset.__getitem__(1)


# +
# img, target = transforms(img, target)

utils.show([v2.functional.to_dtype(img, torch.uint8, scale=True), target])

# +

model.eval()
pred = model([img.to(device)])
# -

img

pred[0]['masks'] = torch.squeeze(pred[0]['masks'])

# +
from utils import show

show([img.to('cpu'), pred[0]], seg_mask=False)

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
from torchvision import tv_tensors
import sys

sys.path.insert(1, '../src/')
from torchvision import tv_tensors
from dataset import ClimbingHoldDataset
from model import get_model_instance_segmentation_resnet
from model import get_model_instance_segmentation
import torch
from torchvision.models import MobileNet_V3_Large_Weights

# +
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses
    epoch_loss /= i+1
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.flush()
    return epoch_loss


# +
import torchvision.transforms.v2 as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToDtype(torch.uint8, scale=False))
    transforms.append(T.ToDtype(torch.float32, scale=True))
                      
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.1, 
                                        saturation=0.4, hue=0.3))
        transforms.append(T.RandomRotation(180))
        transforms.append(T.RandomPerspective())
        
    transforms.append(T.SanitizeBoundingBoxes())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


mean=torch.tensor([0.485, 0.456, 0.406])
std=torch.tensor([0.229, 0.224, 0.225])
unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

# +
model = get_model_instance_segmentation(2)

# for param in model.named_parameters():
#         print(param[0])


# +
import sys
sys.path.insert(1, '../src/')
import torch
from model import get_model_instance_segmentation
import utils
from engine import evaluate
from dataset import ClimbingHoldDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation_resnet(2)

# freeze backbone layers
# for param in model.parameters():
#     param.requires_grad = False
# #  Unfreeze  the roi_heads
# for param in model.roi_heads.parameters():
#     param.requires_grad = True
# #  Unfreeze region proposal generator 
# for param in model.rpn.parameters():
#      param.requires_grad = True



model.to(device)
# use our dataset and defined transformations
dataset = ClimbingHoldDataset('../data/processed', get_transform(train=True))
dataset_test = ClimbingHoldDataset('../data/processed', get_transform(train=False))



indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:5])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=1,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    collate_fn=utils.collate_fn
)



# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.5, weight_decay=0.0003)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=150,
                                               gamma=0.5)

# let's train it for 10 epochs
num_epochs = 10000

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    # evaluate on the test dataset
    if (epoch % 25 == 0):
        res = evaluate(model, data_loader_test, device)
        res_train = evaluate(model, data_loader, device)

        print(f'Epoch {epoch}: \n mAP = {res}')
        print(f'Training Loss: {loss}')
        print(f'Training mAP = {res_train}')
        print('\n')
        
    lr_scheduler.step()
# print("That's it!")

torch.save(model.state_dict(), 'model.pt')

# +
from torchvision.transforms import v2
import sys
sys.path.insert(1, '../src/')
import utils


dataset = ClimbingHoldDataset('../data/processed', transforms=get_transform(False))


img, target = dataset.__getitem__(25)


# +
# img, target = transforms(img, target)

show_a([unnormalize(v2.functional.to_dtype(img, torch.float, scale=True)), target], seg_mask=True)
# -

model.to('cuda')
model.eval()
pred = model([img.to('cuda')])

pred[0]['masks'] = torch.squeeze(pred[0]['masks'])

# +
from utils import show

show_a([unnormalize(img.to('cpu')), pred[0]], seg_mask=True)


# +
def show_a(sample, bbox = True, seg_mask = False):
    import matplotlib.pyplot as plt
    
    from torchvision.transforms import functional as F
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    image, target = sample
        
    image = F.convert_image_dtype(image, torch.uint8)
    labels = target['labels'].tolist()
    class_to_color = {
        1: 'green',
        2: 'violet',
        3: 'yellow'
    }
    colors = [class_to_color[key] for key in labels]
    masks = target['masks']
    
    if 'scores' in target:
        target['boxes'] = target['boxes'][target['scores']>0.5]
        target['masks'] = torch.squeeze(target['masks'])[target['scores']>0.5]
    
    if bbox == True:
        annotated_image = draw_bounding_boxes(
            image,
            target["boxes"],
            colors=colors,
            width=3,
        )
    if seg_mask == True: 


        ordered_labels, ordered_masks = list(zip(*sorted(zip(labels, masks), key= lambda x: x[0], reverse=True)))
        colors = [class_to_color[key] for key in ordered_labels]
        ordered_masks = torch.stack(list(ordered_masks), dim=0)
        
        threshold = .5
        mask_thresh = torch.where(ordered_masks > threshold, True, False)
        
        annotated_image = draw_segmentation_masks(
            image,
            masks=mask_thresh,
            colors=colors,
            alpha=0.5
        )

    
    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()


import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont

from torchvision.utils import _generate_color_palette


def draw_seg_mask(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
) -> torch.Tensor:

    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    num_masks = masks.size()[0]

    if num_masks == 0:
        warnings.warn("masks doesn't contain any mask. No mask was drawn")
        return image

    out_dtype = torch.uint8
    colors = [
        torch.tensor(color, dtype=out_dtype, device=image.device)
        for color in _parse_colors(colors, num_objects=num_masks)
    ]

    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[:, mask] = (
            img_to_draw[:, mask] * 0.3 + color[:, None] * 0.7
            ).to(torch.uint8)

    out = img_to_draw
    return out.to(out_dtype)


def _parse_colors(
    colors: Union[None, str, Tuple[int, int, int], List[Union[str, Tuple[int, int, int]]]],
    *,
    num_objects: int,
) -> List[Tuple[int, int, int]]:
    """
    Parses a specification of colors for a set of objects.

    Args:
        colors: A specification of colors for the objects. This can be one of the following:
            - None: to generate a color palette automatically.
            - A list of colors: where each color is either a string (specifying a named color) or an RGB tuple.
            - A string or an RGB tuple: to use the same color for all objects.

            If `colors` is a tuple, it should be a 3-tuple specifying the RGB values of the color.
            If `colors` is a list, it should have at least as many elements as the number of objects to color.

        num_objects (int): The number of objects to color.

    Returns:
        A list of 3-tuples, specifying the RGB values of the colors.

    Raises:
        ValueError: If the number of colors in the list is less than the number of objects to color.
                    If `colors` is not a list, tuple, string or None.
    """
    if colors is None:
        colors = _generate_color_palette(num_objects)
    elif isinstance(colors, list):
        if len(colors) < num_objects:
            raise ValueError(
                f"Number of colors must be equal or larger than the number of objects, but got {len(colors)} < {num_objects}."
            )
    elif not isinstance(colors, (tuple, str)):
        raise ValueError("`colors` must be a tuple or a string, or a list thereof, but got {colors}.")
    elif isinstance(colors, tuple) and len(colors) != 3:
        raise ValueError("If passed as tuple, colors should be an RGB triplet, but got {colors}.")
    else:  # colors specifies a single color for all objects
        colors = [colors] * num_objects

    return [ImageColor.getrgb(color) if isinstance(color, str) else color for color in colors]


# -

torch.save(model.state_dict(), 'model.pt')

model = get_model_instance_segmentation_resnet(2)
model.load_state_dict(torch.load('model.pt'))
model.to('cuda')
model.eval()

# +
from engine import evaluate
dataset = ClimbingHoldDataset('../data/processed', get_transform(train=False))



# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=1,
    collate_fn=utils.collate_fn
)


res = evaluate(model, data_loader, 'cuda')

print(f'Epoch {epoch}: \n mAP = {res}')


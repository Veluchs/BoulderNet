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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import torch
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torchvision.transforms as T


# %%
class ClimbingHoldDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        # load image files, and labels
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images/"))))
        self.labels = list(sorted(os.listdir(os.path.join(root_dir, "labels/"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load image and mask
        img_path = os.path.join(self.root_dir, "images/", self.imgs[idx])
        labels_path = os.path.join(self.root_dir, "labels/", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        masks, class_labels = self.labels_to_masks(labels_path, img)
        num_instances = len(masks)
        boxes = []
        for mask in masks:
            boxes.append(self.get_bounding_box(mask))

        # convert everything to Tensors

        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        class_labels = torch.as_tensor(np.array(class_labels), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])).detach().clone()
        
        img = T.functional.pil_to_tensor(img)

        iscrowd = torch.zeros((num_instances,), dtype=torch.int64)

        # return as target dictionary

        target = {}
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def get_bounding_box(self, mask):
        """This function computes the bounding box of a given mask"""

        nonzero_indices = np.nonzero(mask)

        xmin = np.min(nonzero_indices[1])
        xmax = np.max(nonzero_indices[1])
        ymin = np.min(nonzero_indices[0])
        ymax = np.max(nonzero_indices[0])

        return [xmin, ymin, xmax, ymax]

    def labels_to_masks(self, label_path, image) -> np.array:
        """This function computes masks from given polygon labels."""

        label = open(label_path)
        lines = label.readlines()
        width, height = image.size

        masks = []
        class_labels = []

        for line in lines:
            class_label = int(line[0])  # TODO what about multiple digits
            polygon = np.fromstring(line[2:], sep=' ')
            polygon_coordinates = [
                (int(polygon[2*i] * width), int(polygon[2*i + 1] * height))
                for i in range(int(len(polygon)/2))
                ]
            # create empty image with size of image
            mask = Image.new('L', (width, height), 0)
            # draw mask on image
            ImageDraw.Draw(mask).polygon(polygon_coordinates,
                                         outline=1,
                                         fill=1,
                                        )
            masks.append(np.array(mask))
            class_labels.append(class_label)

        return masks, class_labels


# %% [markdown]
# ## Visualization Tools

# %%
def show(sample, bbox = True, seg_mask = False):
    import matplotlib.pyplot as plt
    
    from torchvision.transforms import functional as F
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    image, target = sample
        
    image = F.convert_image_dtype(image, torch.uint8)
    labels = target['labels'].tolist()
    class_to_color = {
        0: 'green',
        1: 'violet',
        2: 'yellow'
    }
    colors = [class_to_color[key] for key in labels]
    masks = target['masks']
    masks = masks.to(torch.bool)
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
        annotated_image = draw_segmentation_masks(
            image,
            masks=ordered_masks,
            colors=colors,
            # alpha=0.1
        )

    
    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()

# %%
ds = ClimbingHoldDataset('../data/')

# %%
a = ds[0]

# %%
show(a, bbox=False, seg_mask=True)

# %%
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


# %%

# %%

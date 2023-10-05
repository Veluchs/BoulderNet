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
from PIL import Image, ImageDraw
import numpy as np


# %%
class ClimbingHoldDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        # load image files, and labels
        self.imgs = list(sorted(os.path.join(root_dir, "images/")))
        self.labels = list(sorted(os.path.join(root_dir, "labels/")))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # load image and mask
        img_path = os.path.join(self.root_dir, "images/", self.imgs[idx])
        labels_path = os.path.join(self.root_dir, "labels/", self.labels[idx])

        img = Image.open(img_path).convert("RGB")

        masks, class_labels = self.labels_to_masks(labels_path, img_path)
        num_instances = len(masks)
        boxes = []
        for mask in masks:
            boxes.append(self.get_bounding_box(mask))

        # convert everything to Tensors

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = torch.tensor(
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            )
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
        ymin = np.max(nonzero_indices[0])
        ymax = np.max(nonzero_indices[0])

        return [xmin, ymin, xmax, ymax]

    def labels_to_masks(self, label_path, img_path) -> np.array():
        """This function computes masks from given polygon labels."""

        label = open(label_path)
        lines = label.readLines()

        image = Image.open(img_path)
        width, height = image.size
     
        masks = []
        class_labels = []

        for line in lines:
            class_label = int(line[1])  # TODO what about multiple digits
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
# %%

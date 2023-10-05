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
import numpy as np
import torch
from PIL import Image, ImageDraw


# %%
class ClimbingHoldDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        # load image files, and labels
        self.imgs = list(sorted(os.path.join(root_dir, "images/")))
        self.labels = list(sorted(os.path.join(root_dir, "labels/")))

    def __len__(self):
        return len(self.imgs)

    def polygon_to_mask(self, label_path, img_path) -> np.array():
        label = open(label_path)
        lines = label.readLines()

        image = Image.open(img_path)
        width, height = image.size

        # create empty mask with size of original image
        mask_template = np.zeros([width, height])

        mask = Image.new('RGB', mask_template.shape, 0)
        mask_draw = ImageDraw.Draw(mask)

        for i, line in enumerate(lines, 1):
            polygon = np.fromstring(line[2:], sep=' ')
            polygon_coordinates = [
                int(polygon[2*i] * width), int(polygon[2*i + 1] * height)
                for i in range(int(len(polygon)/2))
                ]
            mask_draw.polygon(polygon_coordinates, 
                              outline=(i, 0, 0), 
                              fill=(i, 0, 0)
                             )
            # TODO add overflow for more than 256 holds

        return np.array(mask)


# %%

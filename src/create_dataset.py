import os
import torch
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision import tv_tensors
import torchvision

def split(img, masks, labels):
    '''Splits the img and masks in patches of size KERNEL_SIZE.

    Image patches are returned with their corresponding masks (if nonempty)
    and their labels.
    '''

    KERNEL_SIZE = 256
    STRIDE = 256

    img = torchvision.transforms.functional.pil_to_tensor(img)
    patches = img.unfold(1, KERNEL_SIZE, STRIDE).unfold(2, KERNEL_SIZE, STRIDE)
    patches = patches.reshape(3, -1, KERNEL_SIZE, KERNEL_SIZE)
    patches = patches.permute(1, 0, 2, 3)
    patches = tv_tensors.Image(patches)

    mask_patched = masks.unfold(1, KERNEL_SIZE, STRIDE).unfold(2,
                                                                KERNEL_SIZE,
                                                                STRIDE)
    mask_patched = mask_patched.reshape(masks.shape[0], -1,
                                        KERNEL_SIZE, KERNEL_SIZE)

    targets = []
    for patch_index, image_patch in enumerate(patches):
        patch_masks = []
        patch_labels = []
        for index, mask in enumerate(mask_patched):

            if mask[patch_index].any() == True:
                patch_masks.append(mask[patch_index]) 
                patch_labels.append(labels[index]) # TODO better stacking

        patch_masks = torch.stack(patch_masks)
        patch_target = {
            'masks': tv_tensors.Mask(patch_masks),
            'labels': torch.tensor(patch_labels)
        }
        targets.append(patch_target)

    return patches, targets



IMAGE_RES = 768
IMAGE_PATH = 'data/data-masks/img'
MASK_PATH = 'data/data-masks/masks_instances'

image_list = os.listdir(IMAGE_PATH)

for image_file_name in image_list:
    img = Image.open(os.path.join(IMAGE_PATH, image_file_name)).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = v2.functional.resize(img, size=IMAGE_RES)

    mask_folder = os.path.join(MASK_PATH, image_file_name[:-4])
    mask_list = os.listdir(mask_folder)

    masks = []
    labels = []
    for mask_filename in sorted(mask_list):
        mask = torchvision.transforms.functional.pil_to_tensor(Image.open(os.path.join(mask_folder, mask_filename)))
        mask = v2.functional.resize(mask, size=IMAGE_RES)
        masks.append(mask)
        if 'hold' in mask_filename:
            labels.append(0)
        if 'volume' in mask_filename:
            labels.append(1)
        if 'wall' in mask_filename:
            labels.append(2)

    masks = torch.squeeze(torch.stack(masks))

    patches, targets = split(img, masks, labels)

    transform = T.ToPILImage()

    for i, patch in enumerate(patches):
        datapoint = {
            'image': patch,
            'masks': targets[i]['masks'],
            'labels': targets[i]['labels']
        }
        torch.save(datapoint, f'data/processed/{image_file_name[:-4]}-{i}.pt')
import os
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision import tv_tensors
import torchvision

from pycocotools.coco import COCO

from pycocotools import mask as coco_mask

def uncompressed_rle_to_mask(segm):
    # TODO: use own implementation instead of encoding - decoding
    compressed_rle = coco_mask.frPyObjects(segm['segmentation'], segm['segmentation']['size'][0], segm['segmentation']['size'][1])
    decoded_value = coco_mask.decode(compressed_rle)
    return decoded_value

def split(img, masks, labels):
    '''Splits the img and masks in patches of size KERNEL_SIZE.

    Image patches are returned with their corresponding masks (if nonempty)
    and their labels.

    Args:
        img: a PIL image.
        masks: a tensor containing all masks for img.
        labels: 

    Returns: 
        A tesnor containing the img cut into patches as well as the target list
        containing a dictioniary containing masks and labels for each patch.  

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
            if torch.any(mask[patch_index]).item():
                patch_masks.append(mask[patch_index])
                patch_labels.append(labels[index])  # TODO better stacking
        
        # TODO potentially skip image if no holds
    
        patch_masks = torch.stack(patch_masks)
        patch_target = {
            'masks': tv_tensors.Mask(patch_masks),
            'labels': torch.tensor(patch_labels)
        }
        targets.append(patch_target)

    return patches, targets


IMAGE_RES = 768
IMAGE_PATH = '/workspaces/BoulderNet/data/coco/innsbruck/images'
ANNOTATION_PATH = '/workspaces/BoulderNet/data/coco/innsbruck/annotations/instances.json'

coco = COCO(ANNOTATION_PATH)

img_id_list = coco.getImgIds()

for img_id in img_id_list:
    image_file_name = coco.loadImgs(img_id)[0]['file_name']
    print(image_file_name)
    img = Image.open(os.path.join(IMAGE_PATH, image_file_name)).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = v2.functional.resize(img, size=IMAGE_RES)

    ann_id_list = coco.getAnnIds(imgIds=[img_id])
    ann_list = coco.loadAnns(ann_id_list)


    masks = []
    labels = []
    for annotation in ann_list:
        mask = torch.from_numpy(uncompressed_rle_to_mask(annotation))
        mask = v2.functional.resize(mask.unsqueeze(0), size=IMAGE_RES)

        masks.append(mask)
        labels.append(annotation['category_id'] - 1)

    masks = torch.squeeze(torch.stack(masks))
    patches, targets = split(img, masks, labels)

    transform = T.ToPILImage()

    for i, patch in enumerate(patches):
        datapoint = {
            'image': patch,
            'masks': targets[i]['masks'],
            'labels': targets[i]['labels']
        }
        torch.save(datapoint, f'/workspaces/BoulderNet/data/processed/{image_file_name[:-4]}-{i}.pt')
# %%
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
    print(mask_patched.shape)
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


# %%
IMAGE_RES = 768
IMAGE_PATH = '../data/test'
ANNOTATION_PATH = '../data/test/instances.json'
SAVE_PATH = 'test_coco'

coco = COCO(ANNOTATION_PATH)

img_id_list = coco.getImgIds()
print(img_id_list)
for img_id in img_id_list:
    image_file_name = coco.loadImgs(img_id)[0]['file_name']
    print(image_file_name)
    img = Image.open(os.path.join(IMAGE_PATH, image_file_name)).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = v2.functional.resize(img, size=IMAGE_RES)

    ann_id_list = coco.getAnnIds(imgIds=[img_id])
    ann_list = coco.loadAnns(ann_id_list)
    print(ann_list)

    masks = []
    labels = []
    for annotation in ann_list:
        mask = torch.from_numpy(uncompressed_rle_to_mask(annotation))
        mask = v2.functional.resize(mask.unsqueeze(0), size=IMAGE_RES)

        masks.append(mask)
        labels.append(annotation['category_id'] - 1)

    masks = torch.squeeze(torch.stack(masks))
    print(masks.shape)
    patches, targets = split(img, masks, labels)

# %%
import numpy as np
from torchvision.io import write_jpeg
import json

def save_to_coco(images, targets) -> dict:
    annotations = []
    images_coco = []
    anno_id = 1
    for i, img in enumerate(images):
        print(i)
        write_jpeg(img, f"{SAVE_PATH}/img_{i}.jpeg")
        img_coco = {
            "id": i,  # Use the same identifier as the annotation
            "width": img.shape[1],  # Set the width of the image
            "height": img.shape[2],  # Set the height of the image
            "file_name": f"img_{i}",  # Set the file name of the image
            "license": 1,  # Set the license for the image (optional)
            }
        images_coco.append(img_coco)
        for j in range(targets[i]["masks"].shape[0]):
            bin_mask = np.asfortranarray(targets[i]["masks"][j].numpy())
            rle_mask = coco_mask.encode(bin_mask)
            category = targets[i]["labels"][j]
            rle_mask["counts"] = rle_mask["counts"].decode('ascii')
            anno = {
                "id": anno_id,  # Use a unique identifier for the annotation
                "image_id": i,  # Use the same identifier for the image
                "category_id": category.item(),  # Assign a category ID to the object
                "segmentaion": rle_mask,
                "bbox": coco_mask.toBbox(rle_mask).tolist(),  # Specify the bounding box in the format [x, y, width, height]
                "area": coco_mask.area(rle_mask).tolist(),  # Calculate the area of the bounding box
                "iscrowd": 0,  # Set iscrowd to 0 to indicate that the object is not part of a crowd
            }
            annotations.append(anno)
            anno_id += 1

        
    
    # Create the COCO JSON object
    a = {
        "info": {
            "description": "My COCO dataset",  # Add a description for the dataset
            "url": "",  # Add a URL for the dataset (optional)
            "version": "1.0",  # Set the version of the dataset
            "year": 2024,  # Set the year the dataset was created
            "contributor": "Simon Brunner",  # Add the name of the contributor (optional)
            "date_created": "2024-01-01T00:00:00",  # Set the date the dataset was created
        },
        "licenses": [],  # Add a list of licenses for the images in the dataset (optional)
        "images": images_coco,
        "annotations": annotations,  # Add the list of annotations to the JSON object
        "categories": [
            {"id": 1, "name": "hold"},
            {"id": 2, "name": "volume"},
            {"id": 3, "name": "wall"}
        ],  # Add a list of categories for the objects in the dataset
    }

    with open(SAVE_PATH + "/test.json", "w") as f:
        json.dump(a, f, indent=2)

# %%
save_to_coco(patches, targets)

# %%

torchvision.io.write_jpeg(patches[0], "testw.jpg")

# %%
cc = COCO("test.json")

# %%
cc.getImgIds()

# %%
cc.getAnnIds(1)

# %%
seg = cc.loadAnns(2)[0]['segmentaion']
seg

# %%
coco_mask.decode(seg)

# %%
seg.encode('ascii')

# %%
import numpy as np
def show_bitmap(bitmap):
    img = Image.fromarray(bitmap.astype(np.uint8) * 255, mode='L')
    img.show()
    input("Press Enter to continue...")
    img.close()

# %%




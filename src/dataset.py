import os
import torch
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision import tv_tensors
import torchvision


class ClimbingHoldDataset(torch.utils.data.Dataset):

    IMAGE_RES = 1024

    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        # load image files, and labels
        self.imgs = list(sorted(
            os.listdir(os.path.join(root_dir, "images/"))
            ))
        self.labels = list(sorted(
            os.listdir(os.path.join(root_dir, "labels/"))
            ))

        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        '''images are resized and split into 4 parts'''

        # load image and mask
        img_path = os.path.join(self.root_dir, "images/", self.imgs[idx])
        labels_path = os.path.join(self.root_dir, "labels/", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)

        img = v2.functional.resize(img, size=ClimbingHoldDataset.IMAGE_RES)

        masks, class_labels = self.labels_to_masks(labels_path, img)
        num_instances = len(masks)

        # split image and masks in 12 patches parts

        image_patches, target_patches = self.split(img, masks, class_labels)

        # for patch in target_patches:
        #     boxes = []
        #     for mask in patch['masks']:
        #         boxes.append(self.get_bounding_box(mask))
        #     patch['boxes'] = boxes

        # convert everything to TV_tensors

        boxes = tv_tensors.BoundingBoxes(np.array(boxes), dtype=torch.float32)
        class_labels = torch.as_tensor(
            np.array(class_labels),
            dtype=torch.int64
            )
        masks = tv_tensors.Mask(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (
            (boxes[:, 3] - boxes[:, 1]) *
            (boxes[:, 2] - boxes[:, 0])
            ).detach().clone()

        img = tv_tensors.Image(img)

        iscrowd = torch.zeros((num_instances,), dtype=torch.int64)

        # return as target dictionary

        target = {}
        target["boxes"] = boxes
        target["labels"] = class_labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def split(self, img, masks, labels):
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

        masks = torch.Tensor(np.array(masks))
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

                if mask[patch_index].any() is True:
                    patch_masks.append(mask[patch_index])
                    patch_labels.append(labels[index])

            patch_target = {
                'masks': patch_masks,
                'labels': patch_labels
            }
            targets.append(patch_target)

        return patches, targets

    def get_bounding_box(mask):
        """This function computes the bounding box of a given mask"""

        nonzero_indices = torch.nonzero(mask, as_tuple=True)

        xmin = torch.min(nonzero_indices[1])
        xmax = torch.max(nonzero_indices[1])
        ymin = torch.min(nonzero_indices[0])
        ymax = torch.max(nonzero_indices[0])

        return torch.tensor((xmin, ymin, xmax, ymax))

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
                                         fill=1
                                         )
            masks.append(np.array(mask))
            class_labels.append(class_label)

        return masks, class_labels

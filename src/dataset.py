import os
import torch
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision import tv_tensors
import torchvision


class ClimbingHoldDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        # load image files, and labels
        self.data = list(sorted(
            os.listdir(root_dir)
            ))

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''make sure images are in the appropiate size already'''

        # load image and mask
        data_path = os.path.join(self.root_dir, self.data[idx])

        datapoint = torch.load(data_path)

        labels = datapoint['labels']
        masks = datapoint['masks']
        # split image and masks in 12 patches parts

        target = {}
        boxes = []
        for mask in masks:
            boxes.append(self.get_bounding_box(mask))
        boxes = torch.stack(boxes)

        area = (
            (boxes[:, 3] - boxes[:, 1]) *
            (boxes[:, 2] - boxes[:, 0])
            ).detach().clone()
        
        # remove empty bounding boxes
        for i, ar in enumerate(area):
            if ar == 0: 
                area = torch.cat([area[0:i], area[i+1:]])
                boxes = torch.cat([boxes[0:i], boxes[i+1:]])
                labels = torch.cat([labels[0:i], labels[i+1:]])
                masks = torch.cat([masks[0:i], masks[i+1:]])

        target['boxes'] = tv_tensors.BoundingBoxes(
                            boxes,
                            format=tv_tensors.BoundingBoxFormat.XYXY,
                            canvas_size=(256, 256))
        target['area'] = area
        target['image_id'] = torch.tensor([idx])
        target['masks'] = masks
        num_instances = len(target['masks'])
        iscrowd = torch.zeros((num_instances,), dtype=torch.int64)
        target['iscrowd'] = iscrowd
        target['labels'] = labels

        image = datapoint['image'].float()

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)

        return image, target

    

    def get_bounding_box(self, mask):
        """This function computes the bounding box of a given mask"""

        nonzero_indices = torch.nonzero(mask, as_tuple=True)

        xmin = torch.min(nonzero_indices[1])
        xmax = torch.max(nonzero_indices[1])
        ymin = torch.min(nonzero_indices[0])
        ymax = torch.max(nonzero_indices[0])

        return torch.tensor((xmin, ymin, xmax, ymax))

    # def labels_to_masks(self, label_path, image) -> np.array:
    #     """This function computes masks from given polygon labels."""

    #     label = open(label_path)
    #     lines = label.readlines()
    #     width, height = image.size

    #     masks = []
    #     class_labels = []

    #     for line in lines:
    #         class_label = int(line[0])  # TODO what about multiple digits
    #         polygon = np.fromstring(line[2:], sep=' ')
    #         polygon_coordinates = [
    #             (int(polygon[2*i] * width), int(polygon[2*i + 1] * height))
    #             for i in range(int(len(polygon)/2))
    #             ]
    #         # create empty image with size of image
    #         mask = Image.new('L', (width, height), 0)
    #         # draw mask on image
    #         ImageDraw.Draw(mask).polygon(polygon_coordinates,
    #                                      outline=1,
    #                                      fill=1
    #                                      )
    #         masks.append(np.array(mask))
    #         class_labels.append(class_label)

    #     return masks, class_labels

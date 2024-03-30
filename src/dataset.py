import os
import torch
from torchvision import tv_tensors
from torchvision. transforms.functional import pil_to_tensor
from PIL import Image
from pycocotools import mask as coco_mask


class ClimbingHoldDataset(torch.utils.data.Dataset):

    def __init__(self, root, annFile, transforms):
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root
        self.transforms = transforms

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        return pil_to_tensor(img)

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        '''make sure images are in the appropiate size already'''
        id = self.ids[idx]
        image = self._load_image(id)
        targets = self._load_target(id)

        masks = []
        labels = []
        boxes = []
        area = []
        target = {}
        for annotation in targets:
            mask = torch.from_numpy(coco_mask.decode(annotation['segmentation']))
            masks.append(mask)
            labels.append(annotation['category_id'])
            # bboxesfrom [x, y, w, h] to [x0, y0, x1, y1]
            bbox = annotation['bbox']
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes.append(bbox)
            area.append(annotation['area'])

        # remove empty bounding boxes
        masks = torch.stack(masks)
        area = torch.tensor(area)
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)

        boxes = boxes[area != 0]
        labels = labels[area != 0]
        masks = masks[area != 0]
        area = area[area != 0]

        # remove labels == 2 (boulderwall)
        boxes = boxes[labels != 2]
        masks = masks[labels != 2]
        area = area[labels != 2]
        labels = labels[labels != 2]

        # make volumes to be holds
        for index, label in enumerate(labels):
            if label == 1:
                labels[index] = 0

        # move labels one up to conform with maskrcnn
        for index, label in enumerate(labels):
            labels[index] += 1

        # transform everything to tv_tensor
        target['boxes'] = tv_tensors.BoundingBoxes(
                            boxes,
                            format=tv_tensors.BoundingBoxFormat.XYXY,
                            canvas_size=(256, 256))
        target['area'] = area
        target['image_id'] = torch.tensor([idx])
        target['masks'] = tv_tensors.Mask(
                            masks,
                            dtype=torch.bool
        )
        num_instances = len(target['masks'])
        iscrowd = torch.zeros((num_instances,), dtype=torch.int64)
        target['iscrowd'] = iscrowd
        target['labels'] = labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
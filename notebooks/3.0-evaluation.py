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
                      
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


# +
# test evaluation

import sys
sys.path.insert(1, '../src/')
import torch
from model import get_model_instance_segmentation
import utils
from dataset import ClimbingHoldDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(3)
model.to(device)
# use our dataset and defined transformations
dataset_test = ClimbingHoldDataset('../data/processed', get_transform(train=False))


dataset_test = torch.utils.data.Subset(dataset_test, [0, 1])


data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    collate_fn=utils.collate_fn
)

# move model to the right device
model.to(device)
# -



# +
import time
import torch
from torch import inference_mode
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@inference_mode()
def evaluate(model, dataloader, device):
    '''evaluate model on given data in dataloader
    Args:
        model
        dataloader
        device
    
    Returns:
    '''
    model.eval()
    for images, targets in dataloader:
        images = list(img.to(device) for img in images) # TODO send all at once to gpu
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # start timer and inference
        model_time = time.time()
        predictions = model(images)
        model_time = time.time() - model_time

        metric = MeanAveragePrecision(iou_type="bbox")
        # TODO torchmetrics doesn't support tv_tensors
        for i in range(len(targets)):
            targets[i]['boxes']= torch.Tensor(targets[i]['boxes'])
            
        metric.update(predictions, targets)

        result = metric.compute()
        return result


# -

evaluate(model, data_loader_test, device)





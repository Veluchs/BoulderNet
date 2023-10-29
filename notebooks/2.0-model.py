# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from torchvision import tv_tensors
import sys

sys.path.insert(1, '../src/')
from torchvision import tv_tensors
from dataset import ClimbingHoldDataset
from model import get_model_instance_segmentation
import torch
from torchvision.models import MobileNet_V3_Large_Weights

# +
import torch

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses
    epoch_loss /= i+1
    return epoch_loss


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
        transforms.append(T.RandomRotation(180))
        transforms.append(T.RandomPerspective())
        
    transforms.append(T.SanitizeBoundingBoxes())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


# +
import sys
sys.path.insert(1, '../src/')
import torch
from model import get_model_instance_segmentation
import utils
from engine import evaluate
from dataset import ClimbingHoldDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(2)
model.to(device)
# use our dataset and defined transformations
dataset = ClimbingHoldDataset('/datasets/holds', get_transform(train=True))
dataset_test = ClimbingHoldDataset('/datasets/holds', get_transform(train=False))



indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:5])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-1:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=1,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    collate_fn=utils.collate_fn
)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 5000

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    # evaluate on the test dataset
    if (epoch % 100 == 0):
        res = evaluate(model, data_loader_test, device)
        print(f'Epoch {epoch}: mAP = {res}')
        print(f'Training Loss: {loss}')
        
    if (epoch % 500 == 0):       
        lr_scheduler.step()
# print("That's it!")

# +
# test evaluation

import sys
sys.path.insert(1, '../src/')
import torch
from model import get_model_instance_segmentation
import utils
from engine import evaluate
from dataset import ClimbingHoldDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(3)
model.to(device)
# use our dataset and defined transformations
dataset_test = ClimbingHoldDataset('../datasets/holds', get_transform(train=False))


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

# +
print(device)

evaluate(model, data_loader_test, device)

# +
from torchvision.transforms import v2
sys.path.insert(1, '../src/')
import utils




inf_transforms = v2.Compose([
    v2.ToDtype(torch.uint8, scale=False),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




dataset = ClimbingHoldDataset('/datasets/holds', transforms=get_transform(False))


img, target = dataset.__getitem__(1)


# +
# img, target = transforms(img, target)

utils.show([v2.functional.to_dtype(img, torch.uint8, scale=True), target])

# +

model.eval()
pred = model([img.to(device)])
# -

pred

pred[0]['masks'] = torch.squeeze(pred[0]['masks'])

# +
from utils import show

show([img.to('cpu'), pred[0]], seg_mask=True)
# -

dataset = ClimbingHoldDataset('/datasets/holds', get_transform(train=True))
dataset.train_labels 

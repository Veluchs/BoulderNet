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
from model import get_model_instance_segmentation_resnet
from model import get_model_instance_segmentation
import torch
from torchvision.models import MobileNet_V3_Large_Weights

# +
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

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
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.flush()
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


mean=torch.tensor([0.485, 0.456, 0.406])
std=torch.tensor([0.229, 0.224, 0.225])
unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

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
model = get_model_instance_segmentation_resnet(2)

# freeze backbone layers
# for param in model.parameters():
#     param.requires_grad = False
# #  Unfreeze  the roi_heads
# for param in model.roi_heads.parameters():
#     param.requires_grad = True
# #  Unfreeze region proposal generator 
# for param in model.rpn.parameters():
#      param.requires_grad = True



model.to(device)
# use our dataset and defined transformations
dataset = ClimbingHoldDataset('../data/processed', get_transform(train=True))
dataset_test = ClimbingHoldDataset('../data/processed', get_transform(train=False))



indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:20])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-20:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=12,
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



# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.5, weight_decay=0.0003)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=50,
                                               gamma=0.8)

# let's train it for 10 epochs
num_epochs = 10000

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # update the learning rate
    # evaluate on the test dataset
    if (epoch % 25 == 0):
        res = evaluate(model, data_loader_test, device)
        res_train = evaluate(model, data_loader, device)

        print(f'Epoch {epoch}: \n mAP = {res}')
        print(f'Training Loss: {loss}')
        print(f'Training mAP = {res_train}')
        print('\n')
        
    lr_scheduler.step()
# print("That's it!")

torch.save(model.state_dict(), 'model.pt')

# +
from torchvision.transforms import v2
import sys
sys.path.insert(1, '../src/')
from dataset import ClimbingHoldDataset
import utils


dataset = ClimbingHoldDataset('../data/processed', transforms=get_transform(False))


img, target = dataset.__getitem__(90)


# +
# img, target = transforms(img, target)

show_a([unnormalize(v2.functional.to_dtype(img, torch.float, scale=True)), target], seg_mask=True, bbox=True)

# +
model.to('cuda')
model.eval()
pred = model([img.to('cuda')])
pred[0]['masks'] = torch.squeeze(pred[0]['masks'])
from utils import show

show_a([unnormalize(img.to('cpu')), pred[0]], seg_mask=True, bbox=True)
# -



torch.save(model.state_dict(), 'model.pt')

model = get_model_instance_segmentation_resnet(2)
model.load_state_dict(torch.load('model.pt'))
model.to('cuda')
model.eval()

datapoint = torch.load('/notebooks/data/processed/20230827_151815-0.pt')

datapoint.keys()

# BoulderNet - A MaskRCNN to detect and segment climbing holds

Python script used to finetune a pretrained MaskRCNN netowork with different backbones. Trained on a custom Dataset containing Fotos of different Climbing Walls in different gyms.

So far the 


## Dataset

The dataset consist of annotated (segmentation masks) images of indoor boulder walls in the COCO format. It consists of the following categories:

- hold: All climbing holds. 
- *volume: All box shaped volumes. These are used to create different shapes on the wall. They typically have bolt holes to allow for additional holds to be mounted on them.
- *wall: The climbing wall on top of which all holds and volumes are mounted.

* Currently not used in training the model.

There is still ongoing work to include more images. The goal is to have roughly 5000 annotated holds.

It will be made public once it is deemed extensive enough.

## Model

## TODO

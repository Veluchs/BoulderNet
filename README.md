# BoulderNet - A MaskRCNN to detect and segment climbing holds

Python script used to finetune a pretrained MaskRCNN netowork with different backbones. Trained on a custom Dataset containing Fotos of different Climbing Walls in different gyms.
The model returns instance masks of climbing holds.

So far the predictions are still prone to errors:
<table>
  <tr>
    <th>Annotation</th>
    <td>
<img src="https://github.com/Veluchs/BoulderNet/assets/135350576/90f6384b-f083-44be-9348-1ef9321b7500" width=200>
    </td>
    <td>
<img src="https://github.com/Veluchs/BoulderNet/assets/135350576/d120b9c0-506f-412b-bf33-5b27c795370d" width=200>
    </td>
    <td>
      <img src="https://github.com/Veluchs/BoulderNet/assets/135350576/bd424461-9feb-4429-b696-2d73952fbd6f" width=200>
      <td>
             <img src="https://github.com/Veluchs/BoulderNet/assets/135350576/c0178fa3-f068-4f65-b734-9565b4845e77" width=200>
      </td>
  </tr>
  <tr>
        <th>Prediction</th>
    <td>
    <img src="https://github.com/Veluchs/BoulderNet/assets/135350576/db972464-001b-4ffd-92da-582958ac64b4" width=200>
  </td>
    <td>
    <img src="https://github.com/Veluchs/BoulderNet/assets/135350576/5f65dedc-0e57-46a0-ae52-5c3e0ae86536" width=200>
  </td>
    <td>
      <img src="https://github.com/Veluchs/BoulderNet/assets/135350576/7ef56807-4091-426f-9de8-8bea68f19d7c" width=200>
    </td>
    <td>
      <img src="https://github.com/Veluchs/BoulderNet/assets/135350576/b551376f-0a78-4a08-bad3-dd8b11270cca" width=200>
    </td>
  </tr>
</table>



## Dataset

The dataset consist of annotated (segmentation masks) images of indoor boulder walls in the COCO format. It consists of the following categories:

- hold: All climbing holds. 
- *volume: All box shaped volumes. These are used to create different shapes on the wall. They typically have bolt holes to allow for additional holds to be mounted on them.
- *wall: The climbing wall on top of which all holds and volumes are mounted.

* Currently not used in training the model.


### Dataset Makeup

| Images  | Holds | Volumes |
| ------------- | ------------- | -------------|
| 17  | 2966 | 80  |


There is still ongoing work to include more images. The goal is to have roughly 5000 annotated holds.

The plan is to publish the dataset once it is deemed extensive enough.

## Model

So far the MaskRCNN model has been trained with a ResNet backbone. The goal is to evaluate the performance when using a MobileNet backbone.

## TODO

- [ ] Clean up datapipeline and image preprocessing
- [ ] Add seperate training script for mobilenet backbone
- [ ] More data :)

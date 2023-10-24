from torchvision.models import mobilenet_v3_large
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torchvision.ops import MultiScaleRoIAlign


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    backbone = mobilenet_v3_large(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 960

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2,
    )

    # put the pieces together inside a Faster-RCNN model

    mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                                output_size=14,
                                                                sampling_ratio=2)

    model = MaskRCNN(backbone,
                            num_classes=num_classes,
                            rpn_anchor_generator=anchor_generator,
                            box_roi_pool=roi_pooler,
                            mask_roi_pool=mask_roi_pooler)

    return model
import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    SSD300_VGG16_Weights,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator


def create_retinanet_model(num_classes=3):
    anchor_sizes = (
        (32, 64, 128),
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048),
    )

    # tried this anchors with letterbox and it was bad
    # anchor_sizes = (
    #     (16, 32, 64),
    #     (32, 64, 128),
    #     (64, 128, 256),
    #     (128, 256, 512),
    #     (256, 512, 1024),
    # )

    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        anchor_generator=anchor_generator,
        num_classes=num_classes,
        score_thresh=0.1,
        nms_thresh=0.5,
        detections_per_img=50,
        topk_candidates=1000,
    )

    return model


def load_retinanet_model(model_name="moorhuhn_retinanet.pth"):
    model = create_retinanet_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, device


# test
def create_ssd_model(num_classes=3):
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    num_anchors = model.anchor_generator.num_anchors_per_location()

    for i, module in enumerate(model.head.classification_head.module_list):
        in_channels = module.in_channels
        new_module = torch.nn.Conv2d(
            in_channels,
            num_classes * num_anchors[i],
            kernel_size=3,
            padding=1,
        )
        model.head.classification_head.module_list[i] = new_module

    model.num_classes = num_classes
    return model

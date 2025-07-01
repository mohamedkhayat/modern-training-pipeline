import torch.nn as nn
from models.model_avgpool import CNN_AVG_POOL
from models.model_fc import CNN_FC
import os
import albumentations as A
import cv2
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
    convnext_small,
    ConvNeXt_Small_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,
    convnext_large,
    ConvNeXt_Large_Weights,
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
    resnext101_32x8d,
    ResNeXt101_32X8D_Weights,
    resnext101_64x4d,
    ResNeXt101_64X4D_Weights,
    regnet_y_16gf,
    RegNet_Y_16GF_Weights,
)

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

SUPPORTED_MODELS = [
    "cnn_avg",
    "cnn_fc",
    "resnet50",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "regnet",
]


def get_transforms(weights, mean=None, std=None):
    if weights:
        tv_preset = weights.transforms()
        crop_sz = tv_preset.crop_size[0]
        resize_sz = tv_preset.resize_size[0]
        mean = tv_preset.mean
        std = tv_preset.std
        interp = cv2.INTER_LINEAR

        for transform in tv_preset.transforms:
            if hasattr(transform, "interpolation"):
                if transform.interpolation == 2:
                    interp = cv2.INTER_LINEAR
                elif transform.interpolation == 3:
                    interp = cv2.INTER_CUBIC
                elif transform.interpolation == 1:
                    interp = cv2.INTER_NEAREST
                break
    else:
        crop_sz = 224
        resize_sz = 256
        mean = mean
        std = std
        interp = cv2.INTER_LINEAR

    transforms = {
        "train_hard": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(crop_sz, crop_sz), scale=(0.8, 1.0), interpolation=interp
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.CoarseDropout(
                    num_holes_range=(1, 3),
                    hole_height_range=(0.05, 0.2),
                    hole_width_range=(0.05, 0.2),
                    fill=0,
                    p=0.6,
                ),
                A.GaussianBlur(
                    sigma_limit=(0.2, 0.5),
                    blur_limit=0,
                    p=0.8,
                ),
                A.ToGray(method="weighted_average", p=0.1),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2(),
            ],
        ),
        "test": A.Compose(
            [
                A.Resize(resize_sz, resize_sz, interpolation=interp),
                A.CenterCrop(crop_sz, crop_sz),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2(),
            ],
        ),
    }
    return transforms, mean, std


def freeze_backbone(backbone, startpoint):
    grad = False
    for param in backbone.parameters():
        param.requires_grad = grad

    for name, param in backbone.named_parameters():
        if startpoint in name:
            grad = True
        param.requires_grad_(grad)

    return backbone


def get_model(cfg, device, mean, std, out_size):
    if cfg.architecture not in SUPPORTED_MODELS:
        raise ValueError(f"Unkown model : {cfg.architecture}")

    elif cfg.architecture == "cnn_avg":
        backbone = CNN_AVG_POOL(
            out_size, cfg.model.last_filter_size, cfg.model.dropout, cfg.model.alpha
        ).to(device)
        transforms, mean, std = get_transforms(None, mean, std)
        return backbone, transforms, mean, std

    elif cfg.architecture == "cnn_fc":
        backbone = CNN_FC(
            cfg.model.input_size,
            cfg.model.hidden_size,
            out_size,
            cfg.model.dropout,
            cfg.model.last_filter_size,
        ).to(device)
        transforms, mean, std = get_transforms(None, mean, std)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT
        backbone = efficientnet_v2_m(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT
        backbone = efficientnet_v2_s(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_l":
        weights = EfficientNet_V2_L_Weights.DEFAULT
        backbone = efficientnet_v2_l(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT
        backbone = convnext_small(weights=weights)
        backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "convnext_base":
        weights = ConvNeXt_Base_Weights.DEFAULT
        backbone = convnext_base(weights=weights)
        backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "convnext_large":
        weights = ConvNeXt_Large_Weights.DEFAULT
        backbone = convnext_large(weights=weights)
        backbone.classifier[2] = nn.Linear(backbone.classifier[2].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnext50_32x4d":
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        backbone = resnext50_32x4d(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnext101_32x8d":
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        backbone = resnext101_32x8d(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnext101_64x4d":
        weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        backbone = resnext101_64x4d(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "regnet":
        weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        backbone = regnet_y_16gf(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
)
from model import CNN
import albumentations as A
import cv2

supported_models = ["cnn", "resnet50", "efficientnet_v2_m", "efficientnet_v2_s"]


def get_transforms(weights):
    if weights:
        tv_preset = weights.transforms()
        crop_sz = tv_preset.crop_size[0]
        resize_sz = tv_preset.resize_size[0]
        mean = tv_preset.mean
        std = tv_preset.std
        interp = cv2.INTER_LINEAR

    else:
        crop_sz = 224
        resize_sz = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        interp = cv2.INTER_LINEAR

    transforms = {
        "train": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(crop_sz, crop_sz), scale=(0.8, 1.0), interpolation=interp
                ),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.CoarseDropout(
                    num_holes_range=(4, 8),
                    hole_height_range=(0.2, 0.3),
                    hole_width_range=(0.2, 0.3),
                    fill="random_uniform",
                    p=0.8,
                ),
                A.GaussianBlur(
                    sigma_limit=(0.2, 0.5),
                    blur_limit=0,
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std),
                A.ToTensorV2(),
            ]
        ),
        "test": A.Compose(
            [
                A.Resize(resize_sz, resize_sz, interpolation=interp),
                A.CenterCrop(crop_sz, crop_sz),
                A.Normalize(mean=mean, std=std),
                A.ToTensorV2(),
            ]
        ),
    }
    return transforms


def freeze_backbone(backbone, startpoint):
    grad = False
    for param in backbone.parameters():
        param.requires_grad = grad

    for name, param in backbone.named_parameters():
        if startpoint in name:
            grad = True
        param.requires_grad_(grad)

    return backbone


def get_model(cfg, device):
    if cfg.architecture not in supported_models:
        raise ValueError(f"Unkown model : {cfg.architechture}")

    elif cfg.architecture == "cnn":
        backbone = CNN(cfg.model.hidden_size, cfg.model.out_size, cfg.model.dropout).to(
            device
        )
        transforms = get_transforms(None)
        return backbone, transforms

    elif cfg.architecture == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, cfg.model.out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms

    elif cfg.architecture == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT
        backbone = efficientnet_v2_m(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(
            backbone.classifier[1].in_features, cfg.model.out_size
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms

    elif cfg.architecture == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT
        backbone = efficientnet_v2_s(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(
            backbone.classifier[1].in_features, cfg.model.out_size
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms

import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
)
from models.model_avgpool import CNN_AVG_POOL
from models.model_fc import CNN_FC
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
import cv2

supported_models = [
    "cnn_avg",
    "cnn_fc",
    "resnet50",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
]


def get_transforms(weights, mean=None, std=None, seed=42):
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
        mean = mean
        std = std
        interp = cv2.INTER_LINEAR

    transforms = {
        "train_hard": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(crop_sz, crop_sz), scale=(0.8, 1.0), interpolation=interp
                ),
                A.HorizontalFlip(p=1.0),
                A.Rotate(limit=15, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.8),
                A.CoarseDropout(
                    num_holes_range=(2, 3),
                    hole_height_range=(0.1, 0.2),
                    hole_width_range=(0.1, 0.2),
                    fill=0,
                    p=0.8,
                ),
                A.GaussianBlur(
                    sigma_limit=(0.2, 0.5),
                    blur_limit=0,
                    p=0.8,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2(),
            ],
            seed=seed,
        ),
        "train_med": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(crop_sz, crop_sz), scale=(0.8, 1.0), interpolation=interp
                ),
                A.SquareSymmetry(p=0.5),
                A.ElasticTransform(alpha=50, sigma=5, p=0.3),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2(),
            ],
            seed=seed,
        ),
        "test": A.Compose(
            [
                A.Resize(resize_sz, resize_sz, interpolation=interp),
                A.CenterCrop(crop_sz, crop_sz),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                A.ToTensorV2(),
            ],
            seed=seed,
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
    if cfg.architecture not in supported_models:
        raise ValueError(f"Unkown model : {cfg.architecture}")

    elif cfg.architecture == "cnn_avg":
        backbone = CNN_AVG_POOL(
            out_size, cfg.model.last_filter_size, cfg.model.dropout, cfg.model.alpha
        ).to(device)
        transforms, mean, std = get_transforms(None, mean, std, cfg.seed)
        return backbone, transforms, mean, std

    elif cfg.architecture == "cnn_fc":
        backbone = CNN_FC(
            cfg.input_size,
            cfg.model.hidden_size,
            out_size,
            cfg.model.dropout,
            cfg.model.last_filter_size,
        ).to(device)
        transforms, mean, std = get_transforms(None, mean, std, cfg.seed)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights, seed=cfg.seed)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT
        backbone = efficientnet_v2_m(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights, seed=cfg.seed)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT
        backbone = efficientnet_v2_s(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, out_size)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights, seed=cfg.seed)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

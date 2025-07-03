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
from torchvision.transforms import InterpolationMode

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
    "regnet_y_16gf",
]


def get_transforms(weights, mean=None, std=None):
    """
    Returns a dictionary of albumentations transforms.

    Args:
        weights (torchvision.models.weights.Weights, optional): Pre-trained weights from torchvision.
                                                              If provided, the transforms will be based on the weights' presets.
        mean (list, optional): Mean for normalization. Used if weights are not provided. Defaults to None.
        std (list, optional): Standard deviation for normalization. Used if weights are not provided. Defaults to None.

    Returns:
        dict: A dictionary containing train and validation transforms.
        mean : List.
        std : List.
    """
    if weights:
        tv_preset = weights.transforms()
        crop_sz = tv_preset.crop_size[0]
        resize_sz = tv_preset.resize_size[0]
        mean = tv_preset.mean
        std = tv_preset.std
        interpolation_mode = tv_preset.interpolation

        if interpolation_mode == InterpolationMode.BILINEAR:
            interp = cv2.INTER_LINEAR
        elif interpolation_mode == InterpolationMode.BICUBIC:
            interp = cv2.INTER_CUBIC
        elif interpolation_mode == InterpolationMode.NEAREST:
            interp = cv2.INTER_NEAREST
        else:
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
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    fill=0,
                    p=0.6,
                ),
                A.GaussianBlur(
                    sigma_limit=(0.2, 0.5),
                    blur_limit=0,
                    p=0.7,
                ),
                A.ToGray(num_output_channels=3, method="weighted_average", p=0.2),
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
    """
    Freezes the parameters of a backbone model up to a specified starting point.
    This function iterates through the parameters of the backbone model and sets
    `requires_grad` to `False` for all parameters until the specified `startpoint`
    is encountered in the parameter names. Once the `startpoint` is found, all subsequent
    parameters will have `requires_grad` set to `True`.
    Args:
        backbone (torch.nn.Module): The backbone model whose parameters are to be frozen.
        startpoint (str): The substring to identify the starting point in the parameter names
            where gradients should start being enabled.
    Returns:
        torch.nn.Module: The backbone model with updated `requires_grad` settings for its parameters.
    """

    grad = False
    for param in backbone.parameters():
        param.requires_grad = grad

    for name, param in backbone.named_parameters():
        if startpoint in name:
            grad = True
        param.requires_grad_(grad)

    return backbone


def get_model(cfg, device, mean, std, n_classes):
    """
    Builds and returns a model based on the provided configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.
        device (str): Device to move the model to ("cpu" or "cuda").
        mean (list): Mean for normalization (used if not a pretrained model).
        std (list): Standard deviation for normalization (used if not a pretrained model).
        n_classes (int): Number of output classes.

    Raises:
        ValueError: If the specified model architecture is not supported.

    Returns:
        tuple: A tuple containing the model, transforms, mean, and std.
    """
    if cfg.architecture not in SUPPORTED_MODELS:
        raise ValueError(f"Unkown model : {cfg.architecture}")

    elif cfg.architecture == "cnn_avg":
        backbone = CNN_AVG_POOL(
            n_classes, cfg.model.last_filter_size, cfg.model.dropout, cfg.model.alpha
        ).to(device)
        transforms, mean, std = get_transforms(None, mean, std)
        return backbone, transforms, mean, std

    elif cfg.architecture == "cnn_fc":
        backbone = CNN_FC(
            cfg.model.input_size,
            cfg.model.hidden_size,
            n_classes,
            cfg.model.dropout,
            cfg.model.last_filter_size,
        ).to(device)
        transforms, mean, std = get_transforms(None, mean, std)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT
        backbone = efficientnet_v2_m(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(
            backbone.classifier[1].in_features, n_classes
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT
        backbone = efficientnet_v2_s(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(
            backbone.classifier[1].in_features, n_classes
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "efficientnet_v2_l":
        weights = EfficientNet_V2_L_Weights.DEFAULT
        backbone = efficientnet_v2_l(weights=weights, dropout=cfg.model.dropout)
        backbone.classifier[1] = nn.Linear(
            backbone.classifier[1].in_features, n_classes
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "convnext_small":
        weights = ConvNeXt_Small_Weights.DEFAULT
        backbone = convnext_small(weights=weights)
        backbone.classifier[2] = nn.Linear(
            backbone.classifier[2].in_features, n_classes
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "convnext_base":
        weights = ConvNeXt_Base_Weights.DEFAULT
        backbone = convnext_base(weights=weights)
        backbone.classifier[2] = nn.Linear(
            backbone.classifier[2].in_features, n_classes
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "convnext_large":
        weights = ConvNeXt_Large_Weights.DEFAULT
        backbone = convnext_large(weights=weights)
        backbone.classifier[2] = nn.Linear(
            backbone.classifier[2].in_features, n_classes
        )

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnext50_32x4d":
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        backbone = resnext50_32x4d(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnext101_32x8d":
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        backbone = resnext101_32x8d(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "resnext101_64x4d":
        weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        backbone = resnext101_64x4d(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

    elif cfg.architecture == "regnet":
        weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        backbone = regnet_y_16gf(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

        backbone = freeze_backbone(backbone, cfg.model.startpoint)

        transforms, mean, std = get_transforms(weights)
        backbone = backbone.to(device)
        return backbone, transforms, mean, std

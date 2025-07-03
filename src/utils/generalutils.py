import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import DictConfig
import random
import numpy as np
import torch.optim as optim
from dataset import DS
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    SequentialLR,
    LinearLR,
)


def set_seed(SEED: int) -> torch.Generator:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    return generator


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_device(cfg: DictConfig) -> torch.device:
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using : {device}")
    return device


def get_optimizer(cfg: DictConfig, model: nn.Module) -> Optimizer:
    if cfg.architecture not in ["cnn_fc", "cnn_avg"]:
        head_params_list = []
        backbone_params_list = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            elif name.startswith("fc") or name.startswith("classifier"):
                head_params_list.append(param)
            else:
                backbone_params_list.append(param)

        optimizer = optim.AdamW(
            [
                {"params": head_params_list, "lr": cfg.lr},
                {"params": backbone_params_list, "lr": cfg.lr / cfg.lr_factor},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    return optimizer


def get_loss(cfg, train_ds: DS, device: torch.device, get_class_weights) -> nn.Module:
    class_weights_tensor = None

    if not cfg.do_sample:
        class_weights = get_class_weights(train_ds)
        class_weights_tensor = torch.zeros(train_ds.n_classes)

        for i in range(train_ds.n_classes):
            class_weights_tensor[i] = class_weights.get(i, 1.0)

        class_weights_tensor = class_weights_tensor.to(device)

    loss = nn.CrossEntropyLoss(weight=class_weights_tensor)

    return loss


def get_scheduler(cfg: DictConfig, optimizer: Optimizer) -> SequentialLR:
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
    )

    cosine_scheduler = (
        CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        if cfg.restarts
        else CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6
        )
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )
    return scheduler


def unnormalize(img_tensor, mean, std):
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError(f"img_tensor must be a torch.Tensor, got {type(img_tensor)}")
    device = img_tensor.device
    dtype = img_tensor.dtype
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device, dtype=dtype)
    else:
        mean = mean.clone().detach().to(device=device, dtype=dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=device, dtype=dtype)
    else:
        std = std.clone().detach().to(device=device, dtype=dtype)

    if img_tensor.ndim == 3:  # C, H, W
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif img_tensor.ndim == 4:  # B, C, H, W
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    else:
        raise ValueError(
            f"img_tensor has unexpected number of dimensions: {img_tensor.ndim}"
        )
    return img_tensor * std + mean


def get_last_conv(model: nn.Module) -> nn.Conv2d:
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module

    raise ValueError("No Conv2d layer found -- Grad-CAM needs a conv layer.")


def clear_memory(train_dl: DataLoader, train_ds: DS) -> None:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    del train_dl, train_ds

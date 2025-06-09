import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autocast
from torch.optim import Optimizer
from torch.amp import GradScaler
from omegaconf import DictConfig
from torchmetrics.classification import F1Score, Accuracy
import random
import numpy as np
import torch.optim as optim
from captum.attr import LayerGradCam

def set_seed(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    return generator

def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def train(
    model: nn.Module,
    device: torch.device,
    train_dl: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    optimizer: Optimizer,
    scaler: GradScaler,
):
    """Trains the model for one epoch using mixed precision.

    Args:
        model (nn.Module): Model to be trained.
        device (torch.device): Device (CPU or GPU) to train on.
        train_dl (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        num_classes (int): Number of label classes to be used for F1 score
        optimizer (Optimizer): Optimizer for updating model weights.
        scaler (GradScaler): Gradient scaler for mixed precision training.
    """
    loss = 0.0
    f1 = F1Score("multiclass", num_classes=num_classes, average="weighted").to(device)
    acc = Accuracy(task="multiclass", num_classes=num_classes, average="weighted").to(
        device
    )

    model.train()

    for batch_idx, (x, y) in enumerate(train_dl):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        device_str = str(device).split(":")[0]

        with autocast(device_type=device_str, dtype=torch.float16):
            out = model(x)
            batch_loss = criterion(out, y)

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        f1.update(out, y)
        acc.update(out, y)
        loss += batch_loss.item()

    f1_score = f1.compute()
    accuracy = acc.compute()
    loss = loss / len(train_dl)

    print(f"Train - F1-Score : {f1_score:.4f} - Loss : {loss:.4f}")

    return loss, f1_score, accuracy

def evaluate(
    model: nn.Module,
    device: torch.device,
    test_dl: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    grad_cam: bool = False,
):
    """Evaluates the model on the test dataset.

    Args:
        model (nn.Module): Model to be evaluated.
        device (torch.device): Device (CPU or GPU) to evaluate on.
        test_dl (DataLoader): Test data loader.
        criterion (nn.Module): Loss function.
        num_classes (int): Number of label classes to be used for F1 score

    """

    y_true = []
    y_pred = []

    loss = 0.0
    f1 = F1Score("multiclass", num_classes=num_classes, average="weighted").to(device)
    acc = Accuracy(task="multiclass", num_classes=num_classes, average="weighted").to(
        device
    )

    attributions = None

    if grad_cam:
        try:
            imgs, _ = next(iter(test_dl))
            input_tensor = imgs.to(device)

            target_layer = get_last_conv(model)
            print(f"Grad-CAM target layer: {target_layer}")

            torch.set_grad_enabled(True)
            model.zero_grad()

            guided_gc = LayerGradCam(model, target_layer)

            outputs = model(input_tensor)

            targets = outputs.argmax(dim=1)
            attributions = guided_gc.attribute(
                input_tensor,
                target=targets,
                relu_attributions=True,
            )
            attributions = torch.nn.functional.interpolate(
                attributions,
                size=imgs.shape[2:],  # Target (H, W) of original image
                mode="bilinear",
                align_corners=False,
            ).detach()  # Detach after interpolation

            torch.set_grad_enabled(False)

        except Exception as e:
            print(f"Error computing attributions : {e}")

    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)

            device_str = str(device).split(":")[0]

            with autocast(device_type=device_str, dtype=torch.float16):
                out = model(x)
                batch_loss = criterion(out, y).item()

            loss += batch_loss
            f1.update(out, y)
            acc.update(out, y)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(np.argmax(out.cpu().numpy(), axis=1))

        f1_score = f1.compute()
        accuracy = acc.compute()

    loss = loss / len(test_dl)
    print(f"Eval - F1-Score : {f1_score:.4f} - Loss : {loss:.4f}")

    return loss, f1_score, accuracy, y_true, y_pred, attributions

def get_optimizer(cfg: DictConfig, model: nn.Module):
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
                {"params": backbone_params_list, "lr": cfg.lr / 10},
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

def get_last_conv(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module

    raise ValueError("No Conv2d layer found -- Grad-CAM needs a conv layer.")

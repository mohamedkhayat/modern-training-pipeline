import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autocast
from torch.optim import Optimizer
from torch.amp import GradScaler
from torchmetrics.classification import F1Score, Accuracy
import numpy as np
from captum.attr import LayerGradCam

from utils.generalutils import get_last_conv


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
        optimizer.zero_grad(set_to_none=True)
        x, y = x.to(device), y.to(device)

        device_str = str(device).split(":")[0]

        with autocast(device_type=device_str, dtype=torch.bfloat16):
            out = model(x)
            batch_loss = criterion(out, y)

        scaler.scale(batch_loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        f1.update(out, y)
        acc.update(out, y)
        loss += batch_loss.item()

    f1_score, accuracy = f1.compute(), acc.compute()

    f1.reset()
    acc.reset()

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

    y_true, y_pred = [], []

    loss = 0.0
    f1 = F1Score("multiclass", num_classes=num_classes, average="weighted").to(device)
    acc = Accuracy(task="multiclass", num_classes=num_classes, average="weighted").to(
        device
    )

    attributions = None
    if grad_cam:
        try:
            torch.cuda.empty_cache()

            imgs, _ = next(iter(test_dl))

            input_tensor = imgs[:12].to(device)

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
            ).detach()

            del guided_gc, input_tensor, outputs, targets
            torch.cuda.empty_cache()
            torch.set_grad_enabled(False)

        except Exception as e:
            print(f"Error computing attributions : {e}")
            torch.cuda.empty_cache()
            torch.set_grad_enabled(False)
            attributions = None

    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)

            device_str = str(device).split(":")[0]

            with autocast(device_type=device_str, dtype=torch.bfloat16):
                out = model(x)
                batch_loss = criterion(out, y).item()

            loss += batch_loss
            f1.update(out, y)
            acc.update(out, y)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(np.argmax(out.float().cpu().numpy(), axis=1))

        f1_score, accuracy = f1.compute(), acc.compute()

        f1.reset()
        acc.reset()

    loss = loss / len(test_dl)
    print(f"Eval - F1-Score : {f1_score:.4f} - Loss : {loss:.4f}")

    return loss, f1_score, accuracy, y_true, y_pred, attributions

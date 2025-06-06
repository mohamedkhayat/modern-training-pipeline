import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import FishDataset
from typing import Dict
from torch import autocast
from torch.optim import Optimizer
from torch.amp import GradScaler
from datetime import datetime
import wandb
from omegaconf import OmegaConf
from torchmetrics.classification import F1Score


def make_data_loaders(
    root_dir: str,
    transforms: Dict,
    train_ratio: float,
    batch_size: int,
    generator: torch.Generator,
):
    """Creates training and testing data loaders from a dataset.

    Args:
        root_dir (str): Path to the directory containing the dataset.
        transforms (dict): Dictionary containing 'train' and 'test' transforms.
        train_ratio (float): Proportion of data to use for training.
        batch_size (int): Batch size for the data loaders.
        generator (torch.Generator): Random number generator for reproducibility.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and testing data loaders.
    """
    dataset = FishDataset(root_dir)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_ds, test_ds = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_ds.dataset.transforms = transforms["train"]
    test_ds.dataset.transforms = transforms["test"]

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    return train_dl, test_dl


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
    metric = F1Score("multiclass", num_classes=num_classes, average="weighted").to(device)

    model.train()

    for batch_idx, (x, y) in enumerate(train_dl):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        with autocast(device_type=device, dtype=torch.float16):
            out = model(x)
            batch_loss = criterion(out, y)

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metric.update(out, y)
        loss += batch_loss.item()

    f1_score = metric.compute()
    loss = loss / len(train_dl)

    print(f"Train - F1-Score : {f1_score:.4f} - Loss : {loss:.4f}")

    return loss, f1_score


@torch.no_grad()
def evaluate(
    model: nn.Module,
    device: torch.device,
    test_dl: DataLoader,
    criterion: nn.Module,
    num_classes: int,
):
    """Evaluates the model on the test dataset.

    Args:
        model (nn.Module): Model to be evaluated.
        device (torch.device): Device (CPU or GPU) to evaluate on.
        test_dl (DataLoader): Test data loader.
        criterion (nn.Module): Loss function.
        num_classes (int): Number of label classes to be used for F1 score

    """

    loss = 0.0
    metric = F1Score("multiclass", num_classes=num_classes, average="weighted").to(device)

    model.eval()

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device, dtype=torch.float16):
            out = model(x)
            batch_loss = criterion(out, y).item()

        loss += batch_loss
        metric.update(out, y)

    f1_score = metric.compute()
    loss = loss / len(test_dl)
    print(f"Eval - F1-Score : {f1_score:.4f} - Loss : {loss:.4f}")

    return loss, f1_score


def get_run_name(cfg):
    name = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"_model={cfg.model.name}_lr={cfg.lr}"
    )
    return name


def initwandb(cfg):
    name = get_run_name(cfg)
    run = wandb.init(
        entity="mohamedkhayat025-none",
        project="FishDataset",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    """
        ={
            "learning_rate": cfg.lr,
            "architecture": cfg.model.name,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "train_ratio": cfg.train_ratio,
            "warmup_epochs": cfg.warmup_epochs,
            "hidden_size": cfg.model.hidden_size,
        },
    """

    return run

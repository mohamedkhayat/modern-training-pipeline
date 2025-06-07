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
import random
import numpy as np
import functools 
from pathlib import Path
import os

def get_data_samples(root_dir):
    root_dir = Path(root_dir)
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    samples = []
    for cls in classes:
        cls_idx = class_to_idx[cls]
        class_path = os.path.join(root_dir, cls)
        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            samples.append((str(img_path), cls_idx))
            
    return samples, classes, class_to_idx

def set_seed(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    
    return generator

def make_dataset(
    root_dir: str,
    train_ratio: float,
    generator: torch.Generator,
    get_stats: bool = False

):
    print("... making dataset ...")
    data_samples, classes, class_to_idx = get_data_samples(root_dir)
    
    dataset_size = len(data_samples)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    
    train_set, test_set = random_split(
        data_samples, [train_size, test_size], generator=generator
    )

    train_ds = FishDataset(train_set, classes, class_to_idx)
    test_ds = FishDataset(test_set, classes, class_to_idx)

    mean,std = None, None
    
    if get_stats:
        mean, std = train_ds.compute_mean_std()

    print("done")
    return train_ds, test_ds, mean, std

def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

        
def make_data_loaders(
    train_ds,
    test_ds,
    transforms,
    batch_size: int,
    generator : torch.Generator
):
    """Creates training and testing data loaders from a dataset.

    Args:
        transforms (dict): Dictionary containing 'train' and 'test' transforms.
        batch_size (int): Batch size for the data loaders.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and testing data loaders.
    """
    print("... making dataloaders ...")
    
    base_seed = generator.initial_seed()
    
    train_ds.transforms = transforms["train"]
    test_ds.transforms = transforms["test"]
    
    worker_init = functools.partial(seed_worker, base_seed=base_seed)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=generator
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=generator
    )

    print("done")
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
    metric = F1Score("multiclass", num_classes=num_classes, average="weighted").to(
        device
    )

    model.train()

    for batch_idx, (x, y) in enumerate(train_dl):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        
        device_str = str(device).split(':')[0]
        
        with autocast(device_type=device_str, dtype=torch.float16):
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
    metric = F1Score("multiclass", num_classes=num_classes, average="weighted").to(
        device
    )

    model.eval()

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)

        device_str = str(device).split(':')[0]

        with autocast(device_type=device_str, dtype=torch.float16):
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
    return run

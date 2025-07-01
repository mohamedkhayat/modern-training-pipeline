from typing import Tuple
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from datetime import datetime
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from torchvision.utils import make_grid
from .generalutils import unnormalize
import matplotlib.cm as cm
import wandb.sdk.wandb_run as Run
from torch.optim.lr_scheduler import SequentialLR


def initwandb(cfg) -> Run:
    """
    Initializes a new wandb run.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        Run: The initialized wandb run object.
    """
    name = get_run_name(cfg)
    run = wandb.init(
        entity=cfg.wandb_entity,
        project="modern-training-pipeline",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def get_run_name(cfg) -> str:
    """
    Generates a run name based on the current date, time, and configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        str: The generated run name.
    """
    name = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"_dataset={cfg.root_dir}"
        + f"_model={cfg.model.name}_lr={cfg.lr}"
    )
    return name


def log_transforms(run, batch, n_images, classes, aug, mean, std) -> None:
    """
    Logs a grid of transformed images to wandb.

    Args:
        run (Run): The current wandb run object.
        batch (tuple): A batch of images and labels.
        n_images (int): The number of images to log.
        classes (list): A list of class names.
        aug (str): The name of the augmentation pipeline.
        mean (torch.Tensor): The mean used for normalization.
        std (torch.Tensor): The standard deviation used for normalization.
    """
    cols = 3
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(rows * 3, cols * 3))
    axes = axes.flatten()

    images, labels = batch
    fig.suptitle(f"{aug} transforms", fontsize=16)

    for ax, img, label in zip(axes, images[:n_images], labels[:n_images]):
        # 1 x 3 x H x W
        img = img.squeeze(0)
        img = unnormalize(img, mean, std).cpu().numpy()
        img = img.transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype("uint8")

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{classes[label]}")

    plt.tight_layout()
    run.log({"transforms visualization": wandb.Image(fig)})
    plt.close(fig)


def log_confusion_matrix(run, y_true, y_pred, classes) -> None:
    """
    Logs a confusion matrix to wandb.

    Args:
        run (Run): The current wandb run object.
        y_true (list): The true labels.
        y_pred (list): The predicted labels.
        classes (list): A list of class names.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.tight_layout()

    run.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)


def log_training_time(run, start_time) -> None:
    """
    Logs the total training time to wandb.

    Args:
        run (Run): The current wandb run object.
        start_time (float): The start time of training.
    """
    end_time = time.time()
    elapsed = end_time - start_time
    run.log({"training time ": elapsed})


def log_model_params(run: Run, model: nn.Module) -> None:
    """
    Logs the total and trainable parameters of a model to wandb.

    Args:
        run (Run): The current wandb run object.
        model (nn.Module): The model.
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    run.log({"total parmas": total_params, "trainable params": trainable_params})


def log_gradcam_to_wandb_streamlined(
    wandb_run: Run,
    batch_for_gradcam: Tuple,
    attributions_batch: torch.Tensor,
    mean_norm: torch.tensor,
    std_norm: torch.tensor,
    max_images_to_log: int = 12,
    gradcam_plot_name: str = "Grad-CAM Overlay",
    overlay_alpha: float = 0.5,
    colormap_name: str = "jet",
    images_per_row: int = 4,
) -> None:
    """
    Logs Grad-CAM overlays to wandb.

    Args:
        wandb_run (Run): The current wandb run object.
        batch_for_gradcam (tuple): A batch of images and labels for Grad-CAM.
        attributions_batch (torch.Tensor): The Grad-CAM attributions.
        mean_norm (torch.Tensor): The mean used for normalization.
        std_norm (torch.Tensor): The standard deviation used for normalization.
        max_images_to_log (int, optional): The maximum number of images to log. Defaults to 12.
        gradcam_plot_name (str, optional): The name of the plot. Defaults to "Grad-CAM Overlay".
        overlay_alpha (float, optional): The alpha value for the overlay. Defaults to 0.5.
        colormap_name (str, optional): The name of the colormap. Defaults to "jet".
        images_per_row (int, optional): The number of images per row in the grid. Defaults to 4.
    """
    original_images, _ = batch_for_gradcam
    original_images = original_images[:max_images_to_log]

    heatmaps = attributions_batch[:max_images_to_log]
    heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-8)

    unnormalized_imgs = unnormalize(original_images, mean_norm, std_norm)
    unnormalized_imgs = torch.clamp(unnormalized_imgs, 0, 1)

    if colormap_name:
        cmap = cm.get_cmap(colormap_name)
        colored_heatmaps_list = []
        for i in range(heatmaps.shape[0]):
            heatmap_single_ch_np = heatmaps[i, 0].cpu().numpy()
            colored_heatmap_np = cmap(heatmap_single_ch_np)[:, :, :3]
            colored_heatmaps_list.append(
                torch.from_numpy(colored_heatmap_np).permute(2, 0, 1)
            )
        heatmaps_processed = torch.stack(colored_heatmaps_list)
    else:
        heatmaps_processed = heatmaps.repeat(1, 3, 1, 1)

    orig_grid = make_grid(
        unnormalized_imgs, nrow=images_per_row, padding=2, normalize=True
    ).cpu()
    heat_grid = make_grid(heatmaps_processed, nrow=images_per_row, padding=2).cpu()

    overlay_grid = (1 - overlay_alpha) * orig_grid + overlay_alpha * heat_grid
    overlay_grid = torch.clamp(overlay_grid, 0, 1)

    cols = min(images_per_row, unnormalized_imgs.size(0))
    rows = (unnormalized_imgs.size(0) + cols - 1) // cols

    fig, ax = plt.subplots(1, 1, figsize=(cols * 2, rows * 2))

    ax.imshow(overlay_grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title(gradcam_plot_name, fontsize=14)
    plt.tight_layout()

    plt.tight_layout()
    wandb_run.log({gradcam_plot_name: wandb.Image(fig)})
    plt.close(fig)


def log_metrics(
    run: Run,
    train_f1: float,
    train_loss: float,
    train_acc: float,
    val_f1: float,
    val_loss: float,
    val_acc: float,
    scheduler: SequentialLR,
) -> None:
    """
    Logs training and validation metrics to wandb.

    Args:
        run (Run): The current wandb run object.
        train_f1 (float): Training F1 score.
        train_loss (float): Training loss.
        train_acc (float): Training accuracy.
        val_f1 (float): Validation F1 score.
        val_loss (float): Validation loss.
        val_acc (float): Validation accuracy.
        scheduler (SequentialLR) : The learning rate scheduler.
    """
    run.log(
        {
            "train f1": train_f1,
            "train loss": train_loss,
            "train acc": train_acc,
            "val f1": val_f1,
            "val loss": val_loss,
            "val acc": val_acc,
            "Learning rate": float(f"{scheduler.get_last_lr()[0]:.6f}"),
        }
    )

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


def initwandb(cfg):
    name = get_run_name(cfg)
    run = wandb.init(
        entity="mohamedkhayat025-none",
        project="FishDataset",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def get_run_name(cfg):
    name = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"_dataset={cfg.root_dir}"
        + f"_model={cfg.model.name}_lr={cfg.lr}"
    )
    return name


def log_transforms(run, batch, n_images, classes, aug, mean, std):
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


def log_confusion_matrix(run, y_true, y_pred, classes):
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


def log_training_time(run, start_time):
    end_time = time.time()
    elapsed = end_time - start_time
    run.log({"training time ": elapsed})


def log_model_params(run, model: nn.Module):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    run.log({"total parmas": total_params, "trainable params": trainable_params})


def log_gradcam_to_wandb_streamlined(
    wandb_run,
    batch_for_gradcam,
    attributions_batch: torch.Tensor,
    mean_norm,
    std_norm,
    max_images_to_log=6,
    gradcam_plot_name="Grad-CAM Overlay",
):
    original_images, _ = batch_for_gradcam
    original_images = original_images[:max_images_to_log]

    heatmaps = attributions_batch[:max_images_to_log]

    unnormalized_imgs = unnormalize(original_images, mean_norm, std_norm)

    heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-8)
    heatmaps_3ch = heatmaps.repeat(1, 3, 1, 1)  # Convert to RGB

    orig_grid = make_grid(
        unnormalized_imgs, nrow=max_images_to_log, padding=2, normalize=True
    ).cpu()
    heat_grid = make_grid(heatmaps_3ch, nrow=max_images_to_log, padding=2).cpu()

    overlay_grid = 0.6 * orig_grid + 0.4 * heat_grid

    fig, axes = plt.subplots(1, 1, figsize=(max_images_to_log * 2, 12))

    for ax, grid, title in zip([axes], [overlay_grid], ["Overlays"]):
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    wandb_run.log({gradcam_plot_name: wandb.Image(fig)})
    plt.close(fig)

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


def initwandb(cfg):
    name = get_run_name(cfg)
    run = wandb.init(
        entity=cfg.wandb_entity,
        project="modern-training-pipeline",
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
    max_images_to_log=12,
    gradcam_plot_name="Grad-CAM Overlay",
    overlay_alpha=0.5,
    colormap_name="jet",
    images_per_row=4,
):
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
    run, train_f1, train_loss, train_acc, val_f1, val_loss, val_acc, scheduler
):
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

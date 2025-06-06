import torch
import torch.nn as nn
import pathlib
from early_stop import EarlyStopping
from utils import make_data_loaders, train, evaluate, initwandb, get_run_name
import torch.optim as optim
from torch.amp import GradScaler
import albumentations as A
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR
import hydra
from omegaconf import DictConfig
from model_factory import get_model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.logwandb:
        run = initwandb(cfg)
        name = run.name
    else:
        name = get_run_name(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using : {device}")

    generator = torch.Generator().manual_seed(42)

    root_dir = pathlib.Path(rf"{cfg.root_dir}")

    model, transforms = get_model(cfg, device)

    train_dl, test_dl = make_data_loaders(
        root_dir, transforms, cfg.train_ratio, cfg.batch_size, generator
    )

    # TODO: either switch dataset, or switch task

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )

    loss = nn.CrossEntropyLoss()
    scaler = GradScaler()
    early_stopper = EarlyStopping(
        patience=10, delta=0.001, path="checkpoints/", name=name
    )

    best_val_f1 = 0.0
    for epoch in range(cfg.epochs):
        print(f"Epoch : {epoch + 1} Learning rate : {scheduler.get_last_lr()[0]:.5f}")
        train_loss, train_f1 = train(model, device, train_dl, loss, cfg.model.out_size, optimizer, scaler)
        val_loss, val_f1 = evaluate(model, device, test_dl, loss, cfg.model.out_size)
        best_val_f1 = max(val_f1, best_val_f1)
        scheduler.step()
        if cfg.logwandb:
            run.log(
                {
                    "train f1": train_f1,
                    "train loss": train_loss,
                    "val f1": val_f1,
                    "val loss": val_loss,
                    "Learning rate": float(f"{scheduler.get_last_lr()[0]:.6f}"),
                }
            )
        if early_stopper(val_f1, model):
            break
    if cfg.logwandb:
        run.log({"best val f1": best_val_f1})
        run.finish()


if __name__ == "__main__":
    main()

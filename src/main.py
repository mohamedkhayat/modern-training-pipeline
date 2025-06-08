import torch
import torch.nn as nn
import pathlib
from early_stop import EarlyStopping
from utils.generalutils import (
    make_data_loaders,
    train,
    evaluate,
    initwandb,
    get_run_name,
    set_seed,
    make_dataset,
    log_confusion_matrix,
    log_transforms,
)
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR
import hydra
from omegaconf import DictConfig
from models.model_factory import get_model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.log:
        run = initwandb(cfg)
        name = run.name
    else:
        name = get_run_name(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using : {device}")

    generator = set_seed(cfg.seed)

    root_dir = pathlib.Path(rf"{cfg.root_dir}")

    train_ds, test_ds, mean, std = make_dataset(
        root_dir,
        cfg.train_ratio,
        cfg.seed,
        cfg.architecture in ["cnn_fc", "cnn_avg", "test"],
    )

    model, transforms = get_model(cfg, device, mean, std)

    train_dl, test_dl = make_data_loaders(
        train_ds, test_ds, transforms, cfg.batch_size, generator, cfg.aug
    )

    log_transforms(
        run, next(iter(train_dl)), cfg.n_images, train_ds.classes, cfg.aug, mean, std
    )
    # TODO: switch dataset
    # TODO: add cross validation
    # TODO: add error analysis

    optimizer = optim.Adam(
        model.parameters(),
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
    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        print(f"Epoch : {epoch + 1} Learning rate : {scheduler.get_last_lr()[0]:.5f}")

        train_loss, train_f1, train_acc = train(
            model, device, train_dl, loss, cfg.model.out_size, optimizer, scaler
        )
        val_loss, val_f1, val_acc, _, _ = evaluate(
            model, device, test_dl, loss, cfg.model.out_size
        )
        best_val_f1 = max(val_f1, best_val_f1)
        best_val_acc = max(val_acc, best_val_acc)

        scheduler.step()

        if cfg.log:
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

        if early_stopper(val_f1, model):
            break

    if cfg.log:
        run.log({"best val f1": best_val_f1})
        run.log({"best val acc": best_val_acc})

        model = early_stopper.get_best_model(model)
        val_loss, val_f1, val_acc, y_true, y_pred = evaluate(
            model, device, test_dl, loss, cfg.model.out_size
        )

        log_confusion_matrix(run, y_true, y_pred, train_ds.classes)

        run.finish()


if __name__ == "__main__":
    main()

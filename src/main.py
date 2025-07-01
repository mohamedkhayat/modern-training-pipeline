import torch
import torch.nn as nn
import pathlib
from early_stop import EarlyStopping
from torch.amp import GradScaler
import hydra
from omegaconf import DictConfig
from models.model_factory import get_model
import time
from utils.generalutils import (
    clear_cache,
    evaluate,
    get_optimizer,
    get_scheduler,
    set_seed,
    train,
)
from utils.data_utils import (
    make_data_loaders,
    make_dataset,
)
from utils.wandb_utils import (
    initwandb,
    get_run_name,
    log_confusion_matrix,
    log_gradcam_to_wandb_streamlined,
    log_model_params,
    log_training_time,
    log_transforms,
)


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

    root_dir = pathlib.Path("data", rf"{cfg.root_dir}")

    train_ds, test_ds, mean, std = make_dataset(
        root_dir,
        cfg.train_ratio,
        cfg.seed,
        cfg.architecture in ["cnn_fc", "cnn_avg"],
        cfg.download_data,
    )

    n_classes = len(train_ds.classes)

    model, transforms, mean, std = get_model(cfg, device, mean, std, n_classes)

    log_model_params(run, model)

    train_dl, test_dl = make_data_loaders(
        train_ds, test_ds, transforms, cfg.batch_size, generator, cfg.aug
    )

    log_transforms(
        run, next(iter(train_dl)), cfg.n_images, train_ds.classes, cfg.aug, mean, std
    )

    # TODO: add cross validation

    optimizer = get_optimizer(cfg, model)

    scheduler = get_scheduler(cfg, optimizer)

    loss = nn.CrossEntropyLoss()
    scaler = GradScaler()

    early_stopper = EarlyStopping(
        patience=cfg.patience, delta=cfg.delta, path="checkpoints/", name=name
    )

    best_val_f1 = 0.0
    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(cfg.epochs):
        print(f"Epoch : {epoch + 1} Learning rate : {scheduler.get_last_lr()[0]:.5f}")

        train_loss, train_f1, train_acc = train(
            model, device, train_dl, loss, n_classes, optimizer, scaler
        )
        val_loss, val_f1, val_acc, *_ = evaluate(
            model, device, test_dl, loss, n_classes
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

        if torch.cuda.is_available():
            clear_cache()

        val_loss, val_f1, val_acc, y_true, y_pred, attributions = evaluate(
            model, device, test_dl, loss, n_classes, grad_cam=True
        )

        log_gradcam_to_wandb_streamlined(
            run, next(iter(test_dl)), attributions, mean, std
        )
        log_training_time(run, start_time)
        log_confusion_matrix(run, y_true, y_pred, train_ds.classes)

        run.finish()


if __name__ == "__main__":
    main()

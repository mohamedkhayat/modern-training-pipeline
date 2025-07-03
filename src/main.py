import pathlib
import os
from early_stop import EarlyStopping
from torch.amp import GradScaler
import hydra
from omegaconf import DictConfig
from models.model_factory import get_model
import time
from utils.train_utils import evaluate, train
from utils.generalutils import (
    get_device,
    get_loss,
    get_optimizer,
    get_scheduler,
    set_seed,
)
from utils.data_utils import make_data_loaders, make_dataset, get_class_weights
from utils.wandb_utils import (
    get_run_and_or_name,
    log_final_report,
    log_metrics,
    log_model_params,
    log_transforms,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    run, name = get_run_and_or_name(cfg)
    device = get_device(cfg)
    generator = set_seed(cfg.seed)

    root_dir = pathlib.Path("data", rf"{cfg.root_dir}")

    train_ds, test_ds, mean, std = make_dataset(
        root_dir,
        cfg.train_ratio,
        cfg.seed,
        cfg.architecture in ["cnn_fc", "cnn_avg"],
        cfg.download_data,
    )

    model, transforms, mean, std = get_model(cfg, device, mean, std, train_ds.n_classes)

    for name, _ in model.named_parameters():
        print(name)

    if cfg.log:
        log_model_params(run, model)

    train_dl, test_dl = make_data_loaders(train_ds, test_ds, transforms, generator, cfg)

    if cfg.log:
        log_transforms(
            run,
            next(iter(train_dl)),
            cfg.n_images,
            train_ds.classes,
            cfg.aug,
            mean,
            std,
        )

    optimizer = get_optimizer(cfg, model)

    scheduler = get_scheduler(cfg, optimizer)

    loss = get_loss(cfg, train_ds, device, get_class_weights)

    scaler = GradScaler()

    early_stopper = EarlyStopping(
        patience=cfg.patience, delta=cfg.delta, path="checkpoints/", name=name
    )

    best_val_f1, best_val_acc = 0.0, 0.0

    start_time = time.time()
    for epoch in range(cfg.epochs):
        print(f"Epoch : {epoch + 1} Learning rate : {scheduler.get_last_lr()[0]:.5f}")

        train_loss, train_f1, train_acc = train(
            model, device, train_dl, loss, train_ds.n_classes, optimizer, scaler
        )

        val_loss, val_f1, val_acc, *_ = evaluate(
            model, device, test_dl, loss, train_ds.n_classes
        )

        best_val_f1 = max(val_f1, best_val_f1)
        best_val_acc = max(val_acc, best_val_acc)

        scheduler.step()

        if cfg.log:
            log_metrics(
                run,
                train_f1,
                train_loss,
                train_acc,
                val_f1,
                val_loss,
                val_acc,
                scheduler,
            )

        if early_stopper(val_f1, model):
            break

    if cfg.log:
        log_final_report(
            run,
            best_val_f1,
            best_val_acc,
            early_stopper,
            train_dl,
            train_ds,
            model,
            device,
            test_dl,
            loss,
            mean,
            std,
            start_time,
        )


if __name__ == "__main__":
    main()

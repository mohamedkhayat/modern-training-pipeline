import torch
import pathlib


class EarlyStopping:
    def __init__(self, patience, delta, path, name):
        self.patience = patience
        self.delta = delta

        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.full_path = path / f"{name}_best.pth"

        self.best_metric = None
        self.counter = 0
        self.earlystop = False

    def __call__(self, val_metric, model):
        if self.best_metric is None:
            self.best_metric = val_metric

        elif val_metric <= self.best_metric + self.delta:
            self.counter += 1
        else:
            self.best_metric = val_metric
            torch.save(model.state_dict(), self.full_path)
            self.counter = 0
            print("saved model weights")

        if self.counter >= self.patience:
            print("early stop triggered")
            self.earlystop = True

        return self.earlystop

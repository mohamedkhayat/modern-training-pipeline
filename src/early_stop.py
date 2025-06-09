import torch
import pathlib


class EarlyStopping:
    def __init__(self, patience, delta, path, name):
        self.patience = patience
        self.delta = delta

        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = name

        self.best_metric = None
        self.counter = 0
        self.earlystop = False

        self.saved_checkpoints = []

    def __call__(self, val_metric, model):
        if self.counter >= self.patience:
            print("early stop triggered")
            self.earlystop = True
            self.cleanup_checkpoints()

        if self.best_metric is None:
            self.best_metric = val_metric

        elif val_metric <= self.best_metric + self.delta:
            self.counter += 1
        else:
            self.best_metric = val_metric
            filename = f"{self.name}_{val_metric:.4f}.pth"
            full_path = self.path / filename
            torch.save(model.state_dict(), full_path)
            self.saved_checkpoints.append((val_metric, full_path))

            self.counter = 0
            print("saved model weights")

        return self.earlystop

    def cleanup_checkpoints(self):
        if not self.saved_checkpoints:  # Check if the list is empty
            print("No checkpoints to clean up.")
            return

        print("cleaning up old checkpoints...")
        best_val, best_path = max(self.saved_checkpoints, key=lambda x: x[0])

        for val, path in self.saved_checkpoints:
            if path != best_path and path.exists():
                try:
                    path.unlink()
                    print(f"deleted {path.name}")
                except Exception as e:
                    print(f"could not delete {path.name}: {e}")

        print(f"kept best model: {best_path.name}")

    def get_best_model(self, model):
        model.eval()

        if len(self.saved_checkpoints) > 0:
            _, best_path = max(self.saved_checkpoints, key=lambda x: x[0])
            model.load_state_dict(torch.load(best_path, weights_only=True))

        return model

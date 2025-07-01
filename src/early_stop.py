import torch
import pathlib


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience, delta, path, name):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            name (str): Name for the checkpoint.
        """
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
        """
        Checks if training should be stopped based on the validation metric.

        Args:
            val_metric (float): The validation metric to monitor.
            model (torch.nn.Module): The model to save if the metric improves.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_metric is None:
            self.best_metric = val_metric
            print("saved model weights")
            self.save_model(model, val_metric)

        elif val_metric <= self.best_metric + self.delta:
            self.counter += 1
        else:
            self.best_metric = val_metric
            self.save_model(model, val_metric)
            self.counter = 0
            print("saved model weights")

        if self.counter >= self.patience:
            print("early stop triggered")
            self.earlystop = True

        return self.earlystop

    def save_model(self, model, val_metric):
        """
        Saves the model checkpoint.

        Args:
            model (torch.nn.Module): The model to save.
            val_metric (float): The validation metric, used for naming the checkpoint file.
        """
        filename = f"{self.name}_{val_metric:.4f}.pth"
        full_path = self.path / filename
        torch.save(model.state_dict(), full_path)
        self.saved_checkpoints.append((val_metric, full_path))

    def cleanup_checkpoints(self):
        """
        Removes all saved checkpoints except for the one with the best validation metric.
        """
        if not self.saved_checkpoints:
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
        """
        Loads the weights of the best model into the provided model object.

        Args:
            model (torch.nn.Module): The model to load the best weights into.

        Returns:
            torch.nn.Module: The model with the loaded best weights.
        """
        self.cleanup_checkpoints()
        model.eval()

        if len(self.saved_checkpoints) > 0:
            _, best_path = max(self.saved_checkpoints, key=lambda x: x[0])
            model.load_state_dict(torch.load(best_path, weights_only=True))

        return model

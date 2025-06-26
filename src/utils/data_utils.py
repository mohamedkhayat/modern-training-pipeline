import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import FishDataset
import functools
from pathlib import Path
import os
from .generalutils import seed_worker

def download_datset():
    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")
    if api_key is None or username is None:
        print("Environment variable 'kaggle_key' and or 'username' is not set!")
        username = str(input("enter username : "))
        api_key = str(input("enter api_key : "))

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_key

    from kaggle import api

    api.authenticate()
    print("download dataset")
    api.dataset_download_files(
        "zlatan599/mushroom1", path="./data", unzip=True
    )

def get_data_samples(root_dir):
    root_dir = Path(root_dir)
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    samples = []
    for cls in classes:
        cls_idx = class_to_idx[cls]
        class_path = os.path.join(root_dir, cls)
        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            samples.append((str(img_path), cls_idx))

    return samples, classes, class_to_idx


def make_dataset(
    root_dir: str,
    train_ratio: float,
    SEED: int,
    get_stats: bool = False,
    translate: bool = False,
):
    print("... making dataset ...")
    data_samples, classes, class_to_idx = get_data_samples(root_dir)
    """
    dataset_size = len(data_samples)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(
        data_samples, [train_size, test_size], generator=generator
    )
    """

    train_set, test_set = train_test_split(
        data_samples,
        train_size=train_ratio,
        stratify=[label for _, label in data_samples],
        random_state=SEED,
    )

    train_ds = FishDataset(train_set, classes, class_to_idx, translate)
    test_ds = FishDataset(test_set, classes, class_to_idx, translate)

    mean, std = None, None

    if get_stats:
        mean, std = train_ds.get_mean_std(root_dir)

    print("... done ...")
    return train_ds, test_ds, mean, std


def make_data_loaders(
    train_ds, test_ds, transforms, batch_size: int, generator: torch.Generator, aug: str
):
    """Creates training and testing data loaders from a dataset.

    Args:
        transforms (dict): Dictionary containing 'train' and 'test' transforms.
        batch_size (int): Batch size for the data loaders.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and testing data loaders.
    """
    print("... making dataloaders ...")

    base_seed = generator.initial_seed()

    train_ds.transforms = transforms[f"train_{aug}"]
    test_ds.transforms = transforms["test"]

    worker_init = functools.partial(seed_worker, base_seed=base_seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=generator,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=generator,
    )

    print("... done ...")
    return train_dl, test_dl

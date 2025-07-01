from collections import Counter
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import DS
import functools
from pathlib import Path
import os
from .generalutils import seed_worker
import pandas as pd
from torch.utils.data import WeightedRandomSampler


def download_dataset():
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
    print("... downloading dataset ...")
    api.dataset_download_files("zlatan599/mushroom1", path="./data", unzip=True)
    print(".. done ...")


def get_data_samples(
    root_dir: str,
) -> Tuple[List[Tuple[str, int]], List[str], Dict[str, int]]:
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


def get_mushroom_data_samples(
    root_dir: str,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[str], Dict[str, int]]:
    root_dir = Path("data")

    train_df = pd.read_csv(root_dir / "train.csv")
    print(train_df.label.value_counts())
    train_df["type"] = "train"

    test_df = pd.read_csv(root_dir / "test.csv")
    test_df["type"] = "test"

    df = pd.concat([train_df, test_df], axis=0)

    paths = df["image_path"]
    labels = df["label"]
    types = df["type"]

    train_samples = []
    test_samples = []
    classes = sorted(list(df["label"].unique()))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    for path, label, type in zip(paths, labels, types):
        path = root_dir / "/".join(path.strip().split("/")[3:])
        if Path(path).suffix.lower() in valid_extensions and path.exists():
            cls_idx = class_to_idx[label]
            if type == "train":
                train_samples.append((str(path), cls_idx))
            else:
                test_samples.append((str(path), cls_idx))
        else:
            print(f"WARNING : {path} is not a valid image, skipping")

    return train_samples, test_samples, classes, class_to_idx


def make_dataset(
    root_dir: str,
    train_ratio: float,
    SEED: int,
    get_stats: bool = False,
    download_data=False,
) -> Tuple[DS, DS, torch.Tensor, torch.Tensor]:
    if download_data:
        download_dataset()

    print("... making dataset ...")
    if root_dir == "merged_dataset":
        (train_set, test_set), classes, class_to_idx = get_mushroom_data_samples(
            root_dir
        )
    else:
        data_samples, classes, class_to_idx = get_data_samples(root_dir)
        train_set, test_set = train_test_split(
            data_samples,
            train_size=train_ratio,
            stratify=[label for _, label in data_samples],
            random_state=SEED,
        )

    train_ds = DS(train_set, classes, class_to_idx)
    test_ds = DS(test_set, classes, class_to_idx)

    mean, std = None, None

    if get_stats:
        mean, std = train_ds.get_mean_std(root_dir)

    print("... done ...")
    return train_ds, test_ds, mean, std


def get_sampler(train_ds: DS) -> WeightedRandomSampler:
    class_counts = Counter()
    for _, target in train_ds:
        class_counts[target] += 1

    class_weights = {cls_idx: 1.0 / cnt for cls_idx, cnt in class_counts.items()}

    sample_weights = [class_weights[target] for _, target in train_ds]

    weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(train_ds), replacement=True
    )
    return sampler


def make_data_loaders(
    train_ds, test_ds, transforms, batch_size: int, generator: torch.Generator, aug: str
) -> Tuple[DataLoader, DataLoader]:
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

    sampler = get_sampler(train_ds)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
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

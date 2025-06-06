from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2


class FishDataset(Dataset):
    """
    A dataset class for loading fish images and their labels.
    """

    def __init__(self, root_dir: str):
        """
        Initializes the FishDataset.

        Args:
            root_dir (str): Path to the root directory containing the dataset.
                             Each subdirectory should be a class name,
                             containing the images for that class.
        """
        super(FishDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []
        self.transforms = None
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            class_path = os.path.join(self.root_dir, cls)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        return img, label

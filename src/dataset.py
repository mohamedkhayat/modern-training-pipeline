from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2
import torch
from PIL import Image
import torchvision.transforms.v2 as v2
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
    
    def compute_mean_std(self):
        # from dinesh2911 https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/32
        print("... calculating dataset mean and std ...")
        # Initialize variables to store cumulative sum of pixel values
        mean = torch.zeros(3)  # Assuming RGB images
        var = torch.zeros(3)
        
        # Define transformation to convert image to tensor
        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        # step I: Mean
        for image_path, _ in self.samples:
            # Open image and convert to tensor
            image = Image.open(image_path)
            image_tensor = to_tensor(image)
            mean += torch.mean(image_tensor, dim=(1, 2))

        mean /= len(self.samples)
        
        # step II: Std-dev
        # first we need mean from step I
        
        for image_path, _ in self.samples:
            # Open image and convert to tensor
            image = Image.open(image_path).convert("RGB")
            image_tensor = to_tensor(image)
            var += torch.mean((image_tensor - mean.unsqueeze(1).unsqueeze(2))**2, dim=(1, 2))

        print("done")
        return mean, torch.sqrt(var / len(self.samples)) 
        
    
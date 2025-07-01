from torch.utils.data import Dataset
import cv2
import torch
from PIL import Image
import torchvision.transforms.v2 as v2
import pathlib
import pickle


class DS(Dataset):
    """
    A dataset class for loading fish images and their labels.
    """

    def __init__(self, samples, classes, class_to_idx):
        """
        Initializes the FishDataset.

        Args:
            root_dir (str): Path to the root directory containing the dataset.
                             Each subdirectory should be a class name,
                             containing the images for that class.
        """
        super(DS, self).__init__()
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples = samples
        self.transforms = None

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

    def get_mean_std(self, root_dir):
        """
        Retrieves the mean and standard deviation of the dataset.

        If the mean and standard deviation have been pre-computed and saved,
        they are loaded from files. Otherwise, they are computed and saved.

        Args:
            root_dir (str): The root directory where the mean and std files are stored.

        Returns:
            tuple: A tuple containing the mean and standard deviation.
        """
        output_dir = pathlib.Path("checkpoints") / root_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        mean_path = output_dir / "mean.pkl"
        std_path = output_dir / "std.pkl"

        try:
            with mean_path.open("rb") as f:
                mean = pickle.load(f)

            with std_path.open("rb") as f:
                std = pickle.load(f)

        except FileNotFoundError as e:
            print(f"Error loading mean and std : {e}")
            mean, std = self.compute_mean_std(root_dir)

        return mean, std

    def compute_mean_std(self, root_dir):
        """
        Computes the mean and standard deviation of the dataset.

        This method calculates the mean and standard deviation of the dataset's images,
        which is useful for normalization. The results are then saved to files.

        Args:
            root_dir (str): The root directory where the computed mean and std should be saved.

        Returns:
            tuple: A tuple containing the computed mean and standard deviation.
        """
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
            image = Image.open(image_path).convert("RGB")
            image_tensor = to_tensor(image)
            mean += torch.mean(image_tensor, dim=(1, 2))

        mean /= len(self.samples)

        # step II: Std-dev
        # first we need mean from step I

        for image_path, _ in self.samples:
            # Open image and convert to tensor
            image = Image.open(image_path).convert("RGB")
            image_tensor = to_tensor(image)
            var += torch.mean(
                (image_tensor - mean.unsqueeze(1).unsqueeze(2)) ** 2, dim=(1, 2)
            )
        std = torch.sqrt(var / len(self.samples))
        print("... done calculating ...")
        print("... saving them ...")

        output_dir = pathlib.Path("checkpoints")
        mean_path = output_dir / root_dir / "mean.pkl"
        std_path = output_dir / root_dir / "std.pkl"

        try:
            with mean_path.open("wb") as f:
                pickle.dump(mean, f)
            with std_path.open("wb") as f:
                pickle.dump(std, f)

            print("... done saving ...")
        except Exception as e:
            print(f"Error saving : {e}")

        return mean, std

# **Modern Deep Learning Training Pipeline for Image Classification**

  

A **flexible and extensible PyTorch-based training pipeline** designed for image classification tasks. This project leverages **Hydra for configuration management**, **Weights & Biases for experiment tracking and hyperparameter sweeps**, and **Albumentations for advanced data augmentation**. It demonstrates training a wide range of modern architectures (including EfficientNet, ConvNeXt, ResNeXt, and RegNet) on the ["Mushroom species recognition" Dataset](https://www.kaggle.com/datasets/zlatan599/mushroom1/data) available on Kaggle.

  

---

  

## **📜 Table of Contents**

- [Features](#-features)

- [Project Structure](#-project-structure)

- [Installation](#-installation)

- [Dataset Setup](#-dataset-setup)

- [Configuration](#-configuration)

- [Training](#-training)

- [Hyperparameter Sweeps with W&B](#-hyperparameter-sweeps-with-wb)

- [Supported Models](#-supported-models)

- [Key Training Components](#-key-training-components)

- [Results & Checkpoints](#-results--checkpoints)

- [To-Do](#-to-do)

- [License](#-license)

- [Acknowledgments](#-acknowledgments)

  

---

  

## **✨ Features**

✔ **Flexible Model Architecture**

✔ **Transfer Learning**

✔ **Class Balancing**

✔ **Hydra for Configuration**

✔ **Weights & Bienses Integration**

✔ **Advanced Data Augmentation**

✔ **Mixed Precision Training**

✔ **Gradient Clipping**

✔ **Grad-CAM Support**

✔ **Learning Rate Scheduling**

✔ **Early Stopping**

✔ **Modular Codebase**

✔ **Comprehensive Metrics**

✔ **Efficient Data Loading**

  

---

  

## **📂 Project Structure**

```
modern-training-pipeline/
├── README.md                # Project documentation
├── LICENSE                  # License information
├── requirements.txt         # Python dependencies
├── conf/                    # Configuration files for Hydra
│   ├── config.yaml          # Main configuration file
│   ├── sweep.yaml           # Sweep configuration for W&B
│   └── model/               # Model-specific configurations
│       ├── cnn.yaml         # Custom CNN configuration
│       ├── efficientnet_v2_s.yaml # EfficientNetV2-S configuration
│       ├── efficientnet_v2_m.yaml # EfficientNetV2-M configuration
│       ├── efficientnet_v2_l.yaml # EfficientNetV2-L configuration
│       ├── resnet50.yaml    # ResNet-50 configuration
│       ├── resnext50_32x4d.yaml # ResNeXt-50 configuration
│       ├── convnext_base.yaml # ConvNeXt-Base configuration
│       └── ...              # Other model configurations
├── data/                    # Dataset directory
│   ├── train.csv            # Training dataset metadata
│   ├── test.csv             # Testing dataset metadata
│   └── merged_dataset/      # Merged dataset for training and validation
│       └── .gitkeep         # Placeholder for empty directories
├── src/                     # Source code
│   ├── dataset.py           # Dataset handling
│   ├── early_stop.py        # Early stopping implementation
│   ├── main.py              # Main training script
│   ├── models/              # Model definitions
│   │   ├── cnn.py           # Custom CNN model
│   │   └── model_factory.py # Factory for loading models
│   └── utils/               # Utility functions
│       ├── data_utils.py    # Data-related utilities
│       ├── general_utils.py # General helper functions
│       └── wandb_utils.py   # Weights & Biases utilities
└── checkpoints/             # Directory for saving model checkpoints
```

  

---

  

## **⚡ Installation**

  

### **1️⃣ Clone the Repository**

```bash

git clone https://github.com/your-username/mohamedkhayat-modern-training-pipeline.git

cd mohamedkhayat-modern-training-pipeline

```

  

### **2️⃣ Set Up a Conda Environment**

  

```bash

conda create --name modern-pipeline python=3.9 -y

conda activate modern-pipeline

```

  

### **3️⃣ Install Dependencies**

  

```bash

pip install -r requirements.txt

```

  

> **Note:** Ensure PyTorch with CUDA support is installed if using GPU. Check the [official PyTorch install guide](https://pytorch.org/get-started/locally/).

  

---

  

## **🍄 Dataset Setup**

  

This pipeline is configured for a custom classification dataset defined by CSV files.

  

1. **Image Directory:** Place all your image files inside the `data/merged_dataset/` directory.

2. **CSV Files:** Provide `train.csv` and `test.csv` in the `data/` directory. These files should contain at least two columns:

* `image_path`: The relative path to an image from the `data/` directory (e.g., `merged_dataset/Amanita muscaria/image_01.jpg`).

* `label`: The string name of the class.

  

**Expected directory layout:**

```
data/
├── train.csv                # Training dataset metadata
├── test.csv                 # Testing dataset metadata
└── merged_dataset/          # Merged dataset for training and validation
    ├── Amanita muscaria/    # Class folder
    │   ├── image_01.jpg     # Example image
    │   └── ...              # Other images
    └── Boletus edulis/      # Another class folder
        └── ...              # Other images
```

  

---

  

## **⚙️ Configuration**

  

Managed with **Hydra**.

  

* **Main config:** `conf/config.yaml`

* **Model-specific configs:** `conf/model/*.yaml`

* **Sweep configs:** `conf/sweep.yaml`

  

> **Note:** Before logging to Weights & Biases, you must set your `wandb_entity` in `conf/config.yaml`.

### **Example overrides:**

```bash
# Train with ConvNeXt-Base and a different learning rate
python src/main.py model=convnext_base lr=0.0005

```

  

---

  

## **🎯 Training**

  

### **1️⃣ Default Training**

  

```bash

python src/main.py

```

  

### **2️⃣ Disable W&B Logging**

  

```bash

python src/main.py log=False

```

  

### **3️⃣ Train with a Different Model**

  

```bash

# Train with ResNeXt

python src/main.py model=resnext50_32x4d

  

# Train with EfficientNetV2-L

python src/main.py model=efficientnet_v2_l batch_size=16

```

  

---

  

## **🧪 Hyperparameter Sweeps with W&B**

  

### **Install/Upgrade W&B**

  

```bash

pip install wandb --upgrade

```

  

### **Login**

  

```bash

wandb login

```

  

### **Initialize Sweep**

  

```bash

wandb sweep conf/sweep.yaml

```

  

### **Run Sweep Agent**

  

```bash

wandb agent <YOUR_SWEEP_ID>

```

  

---

  

## **🤖 Supported Models**

  

The pipeline supports a variety of architectures, easily configurable via Hydra.

  

| Model Family     | Config Name(s)                                           | Description                                                                 |
|------------------|----------------------------------------------------------|-----------------------------------------------------------------------------|
| **Custom CNN**   | `cnn`                                                    | A custom-built CNN with configurable layers, hidden size, and dropout.      |
| **ResNet**       | `resnet50`                                               | Classic ResNet-50 architecture from Torchvision.                            |
| **ResNeXt**      | `resnext50_32x4d`, `resnext101_32x8d`                    | Next-generation ResNet with grouped convolutions.                           |
| **EfficientNet** | `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l` | A family of models balancing accuracy and computational cost.              |
| **ConvNeXt**     | `convnext_small`, `convnext_base`, `convnext_large`      | Modernized CNN-inspired architecture with convolutional blocks.             |
| **RegNet**       | `regnet_y_8gf`, `regnet_y_16gf`                          | Models discovered through Neural Architecture Search (NAS).                 |

**Freezing layers:** controlled via `startpoint` in the model's config file. Layers before the specified `startpoint` are frozen during training.

---

  

## **🧩 Key Training Components**

  

* **Optimizer:** `AdamW`

* **Loss Function:** `CrossEntropyLoss`

* **Scheduler:** `LinearLR` warmup + `CosineAnnealingLR`

* **Class Balancing:** Uses `WeightedRandomSampler` to address imbalanced datasets.
* **Metrics:** `Weighted F1-score` via `TorchMetrics`
* **Early Stopping:** monitors validation F1, saves best model
* **Debugging & Visualization:** Integrated Grad-CAM for model explainability.
* **Transforms:** 

	* **Train:** `RandomResizedCrop`, `Flip`, `RGBShift`, `CoarseDropout`, `Blur`, `Normalize`

	* **Val/Test:** `Resize`, `CenterCrop`, `Normalize`

  

---

  

## **📊 Results & Checkpoints**

  

* **Console Logs:** Epoch-wise loss, F1-score, LR.

* **W&B Dashboard:** Full logs, metrics, system stats, and Grad-CAM visualizations.

* **Saved Models:** In `checkpoints/` with run name and `_best.pth` suffix.

  

---

  

## **📌 To-Do**

  
* [ ] Implement automated testing for utility functions.

  

---

  

## **📄 License**

  

MIT License — see the `LICENSE` file for details.

  

---

  

## **🙌 Acknowledgments**

  

* **Libraries:** PyTorch, Hydra, Weights & Biases, Albumentations, TorchMetrics.

* **Inspiration:** Modern deep learning pipelines and best practices from the community.

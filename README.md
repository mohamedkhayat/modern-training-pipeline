# **Modern Deep Learning Training Pipeline for Image Classification**

A **flexible and extensible PyTorch-based training pipeline** designed for image classification tasks. This project leverages **Hydra for configuration management**, **Weights & Biases for experiment tracking and hyperparameter sweeps**, and **Albumentations for advanced data augmentation**. It demonstrates training various models (Custom CNN, ResNet50, EfficientNetV2-S/M) on the "A Large Scale Fish Dataset".

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
✔ **Hydra for Configuration**  
✔ **Weights & Biases Integration**  
✔ **Advanced Data Augmentation**  
✔ **Mixed Precision Training**  
✔ **Learning Rate Scheduling**  
✔ **Early Stopping**  
✔ **Modular Codebase**  
✔ **Comprehensive Metrics**  
✔ **Efficient Data Loading**

---

## **📂 Project Structure**

```

mohamedkhayat-modern-training-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── conf/
│   ├── config.yaml
│   ├── cnnsweep.yaml
│   ├── sweep.yaml
│   └── model/
│       ├── cnn.yaml
│       ├── efficientnet\_v2\_m.yaml
│       ├── efficientnet\_v2\_s.yaml
│       └── resnet50.yaml
├── data/
│   └── NA\_Fish\_Dataset/
│       └── .gitkeep
├── src/
│   ├── dataset.py
│   ├── early\_stop.py
│   ├── main.py
│   ├── model.py
│   ├── model\_factory.py
│   └── utils.py
└── checkpoints/

````

---

## **⚡ Installation**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/mohamedkhayat-modern-training-pipeline.git
cd mohamedkhayat-modern-training-pipeline
````

### **2️⃣ Set Up a Conda Environment**

```bash
conda create --name fishclf python=3.9 -y
conda activate fishclf
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** Ensure PyTorch with CUDA support is installed if using GPU. Check the [official PyTorch install guide](https://pytorch.org/get-started/locally/).

---

## **🐟 Dataset Setup**

This pipeline uses the "A Large Scale Fish Dataset" from Kaggle.

1. Download it from [Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset).
2. Extract the archive.
3. Move class folders (e.g. `Red Mullet`, `Gilt-Head Bream`, etc.) into `data/NA_Fish_Dataset/`.

**Expected directory layout:**

```
data/
└── NA_Fish_Dataset/
    ├── Black Sea Sprat/
    ├── Gilt-Head Bream/
    └── ...
```

---

## **⚙️ Configuration**

Managed with **Hydra**.

* **Main config:** `conf/config.yaml`
* **Model-specific configs:** `conf/model/*.yaml`
* **Sweep configs:** `conf/sweep.yaml`, `conf/cnnsweep.yaml`

### **Example overrides:**

```bash
# Use custom CNN with dropout
python src/main.py model=cnn batch_size=32 model.dropout=0.25

# Change LR and epochs
python src/main.py lr=0.0005 epochs=30
```

---

## **🎯 Training**

### **1️⃣ Default Training**

```bash
python src/main.py
```

### **2️⃣ Disable W\&B Logging**

```bash
python src/main.py logwandb=False
```

### **3️⃣ Train with Different Model**

```bash
python src/main.py model=efficientnet_v2_s
python src/main.py model=cnn model.hidden_size=1024 model.dropout=0.4
```

---

## **🧪 Hyperparameter Sweeps with W\&B**

### **Install/Upgrade W\&B**

```bash
pip install wandb --upgrade
```

### **Login**

```bash
wandb login
```

### **Initialize Sweep**

```bash
# Pretrained models
wandb sweep conf/sweep.yaml

# Custom CNN
wandb sweep conf/cnnsweep.yaml
```

### **Run Sweep Agent**

```bash
wandb agent <YOUR_SWEEP_ID>
```

---

## **🤖 Supported Models**

| Model               | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `cnn`               | Custom CNN with configurable layers, dropout              |
| `resnet50`          | Torchvision pretrained model with optional layer freezing |
| `efficientnet_v2_s` | Pretrained EfficientNetV2-S                               |
| `efficientnet_v2_m` | Pretrained EfficientNetV2-M                               |

**Freezing layers:** controlled via `startpoint` in config (everything before it is frozen).

---

## **🧩 Key Training Components**

* **Optimizer:** `AdamW`
* **Loss Function:** `CrossEntropyLoss`
* **Scheduler:** `LinearLR` warmup + `CosineAnnealingWarmRestarts`
* **Metrics:** `Weighted F1-score` via `TorchMetrics`
* **Early Stopping:** monitors validation F1, saves best model
* **Transforms:**

  * **Train:** `RandomResizedCrop`, `Flip`, `RGBShift`, `CoarseDropout`, `Blur`, `Normalize`
  * **Val/Test:** `Resize`, `CenterCrop`, `Normalize`

---

## **📊 Results & Checkpoints**

* **Console Logs:** Epoch-wise loss, F1-score, LR.
* **W\&B Dashboard:** Full logs, metrics, system stats.
* **Saved Models:** In `checkpoints/` with run name and `_best.pth` suffix.

---

## **📌 To-Do**

* [ ] Change the dataset or make it a semantic segmentation task

---

## **📄 License**

MIT License — see the `LICENSE` file for details.

---

## **🙌 Acknowledgments**

* **Dataset:** ["A Large Scale Fish Dataset"](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
* **Libraries:** PyTorch, Hydra, Weights & Biases, Albumentations, TorchMetrics
* **Inspiration:** Modern deep learning pipelines and best practices
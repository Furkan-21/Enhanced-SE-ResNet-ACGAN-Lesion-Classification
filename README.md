# Enhancing Skin Lesion Classification with GAN-Based Augmentation and Deep Learning

This repository contains the implementation of a deep learning pipeline designed to improve skin lesion classification using the HAM10000 dataset. This project was developed as a Design Project at Istanbul Technical University (İTÜ).

## 📋 Overview

The classification of skin lesions is often hindered by extreme class imbalance and image artifacts like hair. This project addresses these issues through:

- **Image Preprocessing**: Automated hair removal using black-hat transforms and inpainting.
- **Generative Augmentation**: Using ACGAN and DCGAN to synthesize images for underrepresented classes.
- **Architectural Innovation**: Development of an Enhanced SE-ResNet model to optimize feature recalibration.

## 🏗️ System Architecture

The pipeline is modular, moving from raw data to a balanced, high-performance classifier.

### 1. Data Preprocessing

- **Hair Removal**: Isolates dark, thin structures via black-hat transform and reconstructs the area using an inpainting algorithm.
- **Filtering**: To ensure stable GAN training, the two least represented classes (Dermatofibroma and Vascular lesions) were removed from the augmentation stage.

### 2. GAN-Based Augmentation

- **DCGAN**: Generates general synthetic images to increase overall dataset volume.
- **ACGAN**: Uses class-conditional generation to specifically target and balance minority classes like Melanoma and Actinic keratoses.

## 🔬 SE-ResNet vs. Enhanced SE-ResNet

A core contribution of this project is the refinement of the Squeeze-and-Excitation (SE) integration within the ResNet50 architecture.

### Standard SE-ResNet50

- **Integration Point**: In the baseline SE-ResNet50 used for comparison, the SE block is added only after the final residual block (Layer 4).
- **Mechanism**: It recalibrates the final feature maps once before they reach the global average pooling layer and classification head.

### My Enhanced SE-ResNet50

- **Multi-Stage Integration**: Unlike the standard version, the Enhanced model incorporates SE blocks across all four main stages (Layer 1 through Layer 4) of the ResNet architecture.
- **Residual SE Connections**: A novel residual connection is added after each channel recalibration step. This adds the recalibrated features back to the original input tensors.
- **Preservation of Information**: This strategy allows the network to incorporate attention-enhanced features while strictly preserving the original feature representation, leading to better gradient flow and training stability.

## 📊 Performance Results

The combination of ACGAN-augmented data and the Enhanced SE-ResNet architecture produced the best metrics:

| Model | Accuracy | F1 Macro | F1 Weighted | Precision | Recall |
|-------|----------|----------|-------------|-----------|--------|
| ResNet | 0.67 | 0.57 | 0.71 | 0.53 | 0.71 |
| ResNet + DCGAN | 0.91 | 0.84 | 0.91 | 0.85 | 0.84 |
| ResNet + ACGAN | 0.79 | 0.63 | 0.79 | 0.71 | 0.62 |
| DenseNet | 0.73 | 0.61 | 0.75 | 0.58 | 0.70 |
| DenseNet + DCGAN | 0.85 | 0.74 | 0.84 | 0.73 | 0.76 |
| DenseNet + ACGAN | 0.94 | 0.90 | 0.94 | 0.90 | 0.91 |
| SE-ResNet | 0.77 | 0.68 | 0.78 | 0.66 | 0.71 |
| Enhanced SE-ResNet | 0.79 | 0.66 | 0.80 | 0.65 | 0.67 |
| SE-ResNet + DCGAN | 0.88 | 0.79 | 0.87 | 0.79 | 0.81 |
| SE-ResNet + ACGAN | 0.9610 | 0.9361 | 0.9608 | 0.9416 | 0.9315 |
| SE-DenseNet + ACGAN | 0.9554 | 0.9122 | 0.9543 | 0.9002 | 0.9277 |
| Enhanced SE-ResNet + ACGAN | **0.9723** | **0.9539** | **0.9712** | **0.9583** | **0.9499** |

## 🚀 Demo

Try out the model with your own skin lesion images using our Hugging Face Space:

**[🔗 Hugging Face Space Demo](https://huggingface.co/spaces/Furkan-21/skin-lesion-analyzer)**


## 📁 Repository Structure

```
.
├── archive/              # Original dataset and processed images
├── configs/              # Configuration files for GAN training
├── data/                 # Processed CSV files and data splits
├── models/               # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for experiments
├── results/              # Training results and outputs
├── report.pdf            # Project report document
├── README.md             # Project documentation
└── src/                  # Source code
    ├── dataloader.py     # Data loading utilities
    ├── evaluate.py       # Model evaluation functions
    ├── train.py          # Training scripts
    ├── models.py         # Model architectures
    ├── gan/              # GAN implementations (ACGAN, DCGAN)
    └── utils.py          # Utility functions
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/skin-lesion-classification.git
cd skin-lesion-classification

# Install dependencies
pip install -r requirements.txt
```

## 📝 Usage

### Training the Enhanced SE-ResNet Model

```python
from src.train import train_model
from src.models import SEResNet

# Initialize model
model = SEResNet(num_classes=5, pretrained=True)

# Train with ACGAN-augmented data
model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=150
)
```

### Evaluating the Model

```python
from src.evaluate import evaluate_model

metrics, confusion_matrix, report = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device
)
```

## ✍️ Credits

- **Author**: Furkan ÖZTÜRK
- **Supervisor**: Assoc. Prof. Dr. Nazım Kemal ÜRE
- **Institution**: Istanbul Technical University (İTÜ)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HAM10000 dataset creators
- PyTorch and torchvision communities
- Istanbul Technical University for providing the research environment



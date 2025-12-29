# src/models.py
"""
Model architectures for skin lesion classification.
Includes ConvNeXt and DenseNet variants.
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, densenet201, DenseNet201_Weights, resnet50, ResNet50_Weights
import timm

class SkinLesionConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.model = convnext_tiny(weights=weights)
        num_features = self.model.classifier[2].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class SkinLesionDenseNet(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        self.model = densenet201(weights=DenseNet201_Weights.DEFAULT if pretrained else None)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class SkinLesionHybridNet(nn.Module):
    """
    Hybrid CNN-Transformer model for skin lesion classification.
    Uses a state-of-the-art hybrid model from timm (e.g., CoAtNet).
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True, model_name: str = 'coatnet_0_rw_224'):
        super().__init__()
        # model_name can be 'coatnet_0_rw_224', 'cmt_small', 'mobilevit_s', etc.
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError('Unknown model head for hybrid model')

    def forward(self, x):
        return self.model(x)


class BasicResNet(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


"""
GAN models for HAM10000 skin lesion dataset
Includes:
- DCGAN: Deep Convolutional GAN for single-class generation
- ACGAN: Auxiliary Classifier GAN for conditional multi-class generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(m):
    """
    Custom weights initialization for GAN stability
    Applied to both generator and discriminator
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ===== DCGAN Implementation =====

class DCGANGenerator(nn.Module):
    """
    DCGAN Generator - generates images from random noise
    Optimized for 128x128 images
    """
    def __init__(self, latent_dim=100, channels=3, ngf=64):
        """
        Args:
            latent_dim: Dimension of the latent space
            channels: Number of output channels (3 for RGB)
            ngf: Size of feature maps in generator
        """
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # Input is latent vector z
            nn.ConvTranspose2d(latent_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # State size: (ngf*16) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 32 x 32
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 64 x 64
            
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: (channels) x 128 x 128
        )
        
        # Apply custom weight initialization
        self.apply(weights_init)
        
    def forward(self, z):
        """
        Forward pass
        Args:
            z: Batch of latent vectors [batch_size, latent_dim]
        Returns:
            Generated images [batch_size, channels, 128, 128]
        """
        # Reshape z to be compatible with first ConvTranspose2d layer
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator - classifies images as real/fake
    Optimized for 128x128 images
    """
    def __init__(self, channels=3, ndf=64):
        """
        Args:
            channels: Number of input channels (3 for RGB)
            ndf: Size of feature maps in discriminator
        """
        super(DCGANDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is (channels) x 128 x 128
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 64 x 64
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 32 x 32
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 16 x 16
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 8 x 8
            
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*16) x 4 x 4
            
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
            # Output: 1 x 1 x 1 (logit)
        )
        
        # Apply custom weight initialization
        self.apply(weights_init)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Batch of images [batch_size, channels, 128, 128]
        Returns:
            Logit of real/fake [batch_size, 1]
        """
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


# ===== ACGAN Implementation =====

class ACGANGenerator(nn.Module):
    """
    ACGAN Generator - generates images conditioned on class labels
    Optimized for 128x128 images
    """
    def __init__(self, latent_dim=100, n_classes=5, channels=3, ngf=64):
        """
        Args:
            latent_dim: Dimension of the latent space
            n_classes: Number of classes
            channels: Number of output channels (3 for RGB)
            ngf: Size of feature maps in generator
        """
        super(ACGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.ngf = ngf
        
        # Embedding layer for class conditioning
        self.class_embedding = nn.Embedding(n_classes, latent_dim)
        
        # Initial projection layer
        self.project = nn.Linear(latent_dim * 2, ngf * 16 * 4 * 4)
        
        self.output_layer = nn.Sequential(
            # State size: (ngf*16) x 4 x 4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 32 x 32
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 64 x 64
            
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: (channels) x 128 x 128
        )
        
        # Apply custom weight initialization
        self.apply(weights_init)
        
    def forward(self, z, labels):
        """
        Forward pass
        Args:
            z: Batch of latent vectors [batch_size, latent_dim]
            labels: Batch of class labels [batch_size]
        Returns:
            Generated images [batch_size, channels, 128, 128]
        """
        # Get class embeddings
        class_emb = self.class_embedding(labels)
        
        # Concatenate noise and class embedding
        z_class = torch.cat([z, class_emb], dim=1)
        
        # Project and reshape
        x = self.project(z_class)
        x = x.view(x.size(0), self.ngf * 16, 4, 4)
        
        # Output layer
        return self.output_layer(x)


class ACGANDiscriminator(nn.Module):
    """
    ACGAN Discriminator - classifies images as real/fake and predicts class labels
    Optimized for 128x128 images
    """
    def __init__(self, n_classes=5, channels=3, ndf=64):
        """
        Args:
            n_classes: Number of classes
            channels: Number of input channels (3 for RGB)
            ndf: Size of feature maps in discriminator
        """
        super(ACGANDiscriminator, self).__init__()
        
        self.n_classes = n_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Input is (channels) x 128 x 128
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 64 x 64
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 32 x 32
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 16 x 16
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 8 x 8
            
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*16) x 4 x 4
        )
        
        # Real/Fake output
        self.adv_layer = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # Class prediction output
        self.aux_layer = nn.Sequential(
            nn.Conv2d(ndf * 16, n_classes, 4, 1, 0, bias=False),
        )
        
        # Apply custom weight initialization
        self.apply(weights_init)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Batch of images [batch_size, channels, 128, 128]
        Returns:
            validity: Probability of real/fake [batch_size, 1]
            class_pred: Class predictions [batch_size, n_classes]
        """
        features = self.features(x)
        
        validity = self.adv_layer(features).view(-1)
        class_pred = self.aux_layer(features).view(-1, self.n_classes)
        
        return validity, class_pred
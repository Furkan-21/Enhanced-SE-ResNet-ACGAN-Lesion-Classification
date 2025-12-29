"""
Visualization utilities for GAN training and evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from datetime import datetime


def save_image_grid(images, path, nrow=8, title=None, normalize=True):
    """
    Save a grid of images
    Args:
        images: Tensor of images [batch_size, channels, height, width]
        path: Path to save the image
        nrow: Number of images in each row
        title: Title for the plot
        normalize: Whether to normalize the images to [0, 1]
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert to grid
    grid = make_grid(images, nrow=nrow, normalize=normalize, padding=2)
    
    # Convert to numpy and transpose
    grid = grid.cpu().detach().numpy().transpose((1, 2, 0))
    
    # Create figure
    plt.figure(figsize=(12, 12))
    if title:
        plt.title(title)
    
    # Show grid
    if grid.shape[2] == 1:  # Grayscale
        plt.imshow(grid[:, :, 0], cmap='gray')
    else:  # RGB
        plt.imshow(grid)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_loss_plot(g_losses, d_losses, path, title="Generator and Discriminator Loss"):
    """
    Save a plot of generator and discriminator losses
    Args:
        g_losses: List of generator losses
        d_losses: List of discriminator losses
        path: Path to save the plot
        title: Title for the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_class_samples(generator, device, latent_dim, n_classes, n_samples=10, path=None):
    """
    Generate and save samples for each class from an ACGAN generator
    Args:
        generator: ACGAN generator model
        device: Device to run model on
        latent_dim: Dimension of latent space
        n_classes: Number of classes
        n_samples: Number of samples per class
        path: Path to save the images
    """
    generator.eval()
    
    # Generate samples for each class
    with torch.no_grad():
        for c in range(n_classes):
            # Generate fixed noise
            z = torch.randn(n_samples, latent_dim).to(device)
            
            # Generate labels
            labels = torch.full((n_samples,), c, dtype=torch.long).to(device)
            
            # Generate images
            gen_imgs = generator(z, labels)
            
            # Normalize to [0, 1]
            gen_imgs = (gen_imgs + 1) / 2
            
            # Save grid
            if path:
                class_path = os.path.join(path, f"class_{c}.png")
                save_image_grid(gen_imgs, class_path, nrow=5, 
                               title=f"Class {c} Samples")
    
    generator.train()


def save_dcgan_samples(generator, device, latent_dim, n_samples=64, path=None):
    """
    Generate and save samples from a DCGAN generator
    Args:
        generator: DCGAN generator model
        device: Device to run model on
        latent_dim: Dimension of latent space
        n_samples: Number of samples
        path: Path to save the images
    """
    generator.eval()
    
    # Generate samples
    with torch.no_grad():
        # Generate fixed noise
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # Generate images
        gen_imgs = generator(z)
        
        # Normalize to [0, 1]
        gen_imgs = (gen_imgs + 1) / 2
        
        # Save grid
        if path:
            save_image_grid(gen_imgs, path, nrow=8, 
                           title=f"Generated Samples")
    
    generator.train()


def create_timestamp_folder(base_path, prefix="run"):
    """
    Create a folder with a timestamp
    Args:
        base_path: Base path for the folder
        prefix: Prefix for the folder name
    Returns:
        path: Path to the created folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_path, f"{prefix}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path

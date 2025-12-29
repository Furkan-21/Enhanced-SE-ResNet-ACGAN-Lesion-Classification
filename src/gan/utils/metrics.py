"""
Metrics for evaluating GAN quality
Includes:
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from scipy import linalg
from tqdm import tqdm


class InceptionV3Features(nn.Module):
    """
    Inception v3 model for extracting features for FID/KID calculation
    """
    def __init__(self, device='cuda'):
        super(InceptionV3Features, self).__init__()
        # Load pretrained Inception model
        inception = models.inception_v3(pretrained=True)
        # Remove final classification layer
        self.model = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.model = self.model.to(device)
        self.model.eval()
        
    def forward(self, x):
        """
        Extract features from input images
        Args:
            x: Batch of images [batch_size, 3, height, width]
        Returns:
            features: Batch of feature vectors [batch_size, 2048]
        """
        # Resize if needed
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Ensure in [0, 1]
        if x.min() < 0 or x.max() > 1:
            x = torch.clamp(x, 0, 1)
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        # Extract features
        with torch.no_grad():
            features = self.model(x)
            features = features.squeeze(3).squeeze(2)
        return features


def calculate_activation_statistics(dataloader, model, device='cuda', max_samples=None):
    """
    Calculate mean and covariance of features
    Args:
        dataloader: DataLoader for images
        model: Feature extraction model
        device: Device to run model on
        max_samples: Maximum number of samples to use
    Returns:
        mu: Mean of features
        sigma: Covariance of features
    """
    features = []
    n_samples = 0
    
    for batch in tqdm(dataloader, desc="Calculating statistics"):
        if isinstance(batch, dict):
            images = batch['image']
        else:
            images = batch
            
        if images.shape[1] == 1:  # Convert grayscale to RGB
            images = images.repeat(1, 3, 1, 1)
            
        # Scale from [-1, 1] to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2
            
        images = images.to(device)
        batch_features = model(images).cpu().numpy()
        features.append(batch_features)
        
        n_samples += batch_features.shape[0]
        if max_samples is not None and n_samples >= max_samples:
            break
    
    features = np.concatenate(features, axis=0)
    if max_samples is not None:
        features = features[:max_samples]
        
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet distance between two multivariate Gaussians
    Args:
        mu1, mu2: Mean vectors
        sigma1, sigma2: Covariance matrices
        eps: Small constant for numerical stability
    Returns:
        fid: Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(real_dataloader, fake_dataloader, device='cuda', max_samples=None):
    """
    Calculate FID between real and fake images
    Args:
        real_dataloader: DataLoader for real images
        fake_dataloader: DataLoader for fake images
        device: Device to run model on
        max_samples: Maximum number of samples to use
    Returns:
        fid: Fréchet Inception Distance
    """
    # Initialize feature extractor
    inception = InceptionV3Features(device)
    
    # Calculate statistics for real images
    mu_real, sigma_real = calculate_activation_statistics(
        real_dataloader, inception, device, max_samples)
    
    # Calculate statistics for fake images
    mu_fake, sigma_fake = calculate_activation_statistics(
        fake_dataloader, inception, device, max_samples)
    
    # Calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid


def calculate_fid_from_stats(mu_real, sigma_real, mu_fake, sigma_fake):
    """
    Calculate FID directly from pre-computed statistics
    Args:
        mu_real: Mean of real features
        sigma_real: Covariance of real features
        mu_fake: Mean of fake features
        sigma_fake: Covariance of fake features
    Returns:
        fid: Fréchet Inception Distance
    """
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


def polynomial_kernel(X, Y):
    """
    Polynomial kernel for KID calculation
    Args:
        X, Y: Feature matrices
    Returns:
        Kernel matrix
    """
    dot = np.dot(X, Y.T)
    feature_dim = X.shape[1] # Should be 2048 for InceptionV3
    return (dot / feature_dim + 1) ** 3


def calculate_kid(real_features, fake_features, subsets=100, subset_size=1000):
    """
    Calculate KID between real and fake features
    Args:
        real_features: Features from real images
        fake_features: Features from fake images
        subsets: Number of subsets to use
        subset_size: Size of each subset
    Returns:
        kid_mean: Mean KID
        kid_std: Standard deviation of KID
    """
    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    
    subset_size = min(min(n_real, n_fake), subset_size)
    subsets = min(subsets, n_real // subset_size)
    
    if subsets == 0:
        if n_real >= 10 and n_fake >= 10:
            subset_size = min(n_real, n_fake) // 2
            subsets = 2
        else:
            subset_size = min(n_real, n_fake)
            subsets = 1
    
    kid_values = []
    for _ in range(subsets):
        real_idx = np.random.choice(n_real, subset_size, replace=False)
        fake_idx = np.random.choice(n_fake, subset_size, replace=False)
        
        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]
        
        real_real = polynomial_kernel(real_subset, real_subset)
        real_fake = polynomial_kernel(real_subset, fake_subset)
        fake_fake = polynomial_kernel(fake_subset, fake_subset)
        
        mmd = (np.sum(real_real) - np.sum(np.diag(real_real))) / (subset_size * (subset_size - 1))
        mmd += (np.sum(fake_fake) - np.sum(np.diag(fake_fake))) / (subset_size * (subset_size - 1))
        mmd -= 2 * np.sum(real_fake) / (subset_size ** 2)
        
        mmd = mmd * 1000  # Scaling factor to get a reasonable KID range
        
        kid_values.append(mmd)
    
    return np.mean(kid_values), np.std(kid_values)

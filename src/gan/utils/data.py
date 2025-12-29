"""
Data handling utilities for GAN training
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from collections import Counter


class HAM10000GanDataset(data.Dataset):
    """
    Dataset class for HAM10000 skin lesion images for GAN training
    Includes class-balanced sampling and transforms optimized for GAN training
    """
    def __init__(self, csv_file, img_dir, transform=None, class_name=None, image_size=128):
        """
        Args:
            csv_file: Path to CSV file with image names and labels
            img_dir: Directory with images
            transform: Custom transforms to apply
            class_name: If specified, only load images of this class
            image_size: Size to resize images to
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.image_size = image_size
        
        # Get class names
        self.categories = self.data_frame.columns[1:].tolist()
        
        # Filter by class if specified
        if class_name is not None:
            if class_name not in self.categories:
                raise ValueError(f"Class {class_name} not found in dataset")
            
            # Keep only rows where the specified class has value 1.0
            self.data_frame = self.data_frame[self.data_frame[class_name] == 1.0]
        
        # Extract image names and labels
        self.image_filenames = self.data_frame['image'].values
        
        # For multi-class, convert one-hot to indices
        self.labels = np.argmax(self.data_frame[self.categories].values, axis=1)
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_name = os.path.join(self.img_dir, self.image_filenames[idx] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {'image': image, 'label': label}


def create_class_balanced_loader(csv_file, img_dir, batch_size=64, image_size=128, num_workers=4):
    """
    Create a DataLoader with class-balanced sampling
    Args:
        csv_file: Path to CSV file with image names and labels
        img_dir: Directory with images
        batch_size: Batch size
        image_size: Size to resize images to
        num_workers: Number of workers for DataLoader
    Returns:
        dataloader: DataLoader with class-balanced sampling
        class_weights: Weights for each class (for loss weighting)
        class_counts: Number of samples in each class
    """
    # Create dataset
    dataset = HAM10000GanDataset(csv_file, img_dir, image_size=image_size)
    
    # Count samples per class
    class_counts = Counter(dataset.labels)
    
    # Calculate weights for each sample
    # Weight = 1 / (num_samples_in_class * num_classes)
    weights = torch.zeros(len(dataset))
    for idx, label in enumerate(dataset.labels):
        weights[idx] = 1.0 / (class_counts[label] * len(class_counts))
    
    # Create sampler
    sampler = data.WeightedRandomSampler(weights, len(dataset))
    
    # Create dataloader
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Calculate class weights for loss function
    # Weight = 1 / frequency
    class_weights = torch.zeros(len(class_counts))
    for label, count in class_counts.items():
        class_weights[label] = 1.0 / count
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return dataloader, class_weights, class_counts


def create_single_class_loader(csv_file, img_dir, class_name, batch_size=64, image_size=128, num_workers=4):
    """
    Create a DataLoader for a single class (for DCGAN training)
    Args:
        csv_file: Path to CSV file with image names and labels
        img_dir: Directory with images
        class_name: Class to load
        batch_size: Batch size
        image_size: Size to resize images to
        num_workers: Number of workers for DataLoader
    Returns:
        dataloader: DataLoader for the specified class
    """
    # Create dataset
    dataset = HAM10000GanDataset(csv_file, img_dir, class_name=class_name, image_size=image_size)
    
    # Create dataloader
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_class_distribution(csv_file):
    """
    Get the distribution of classes in the dataset
    Args:
        csv_file: Path to CSV file with image names and labels
    Returns:
        class_counts: Dictionary with class counts
    """
    df = pd.read_csv(csv_file)
    categories = df.columns[1:].tolist()
    
    class_counts = {}
    for category in categories:
        class_counts[category] = int(df[category].sum())
    
    return class_counts


def save_generated_images(generator, device, latent_dim, n_classes, output_dir, samples_per_class=1000, class_specific_counts=None):
    """
    Generate and save images for data augmentation
    Args:
        generator: Trained generator model
        device: Device to run model on
        latent_dim: Dimension of latent space
        n_classes: Number of classes
        output_dir: Directory to save images
        samples_per_class: Default number of samples to generate per class (used if class_specific_counts is None)
        class_specific_counts: Dictionary mapping class indices to number of samples to generate
    """
    generator.eval()
    
    # Create directories
    for c in range(n_classes):
        os.makedirs(os.path.join(output_dir, f"class_{c}"), exist_ok=True)
    
    # Generate samples for each class
    with torch.no_grad():
        for c in range(n_classes):
            # Determine how many samples to generate for this class
            if class_specific_counts is not None and c in class_specific_counts:
                num_samples = class_specific_counts[c]
            else:
                num_samples = samples_per_class
                
            print(f"Generating {num_samples} images for class {c}...")
            
            # Generate in batches for efficiency
            batch_size = 16
            for batch_start in range(0, num_samples, batch_size):
                # Calculate actual batch size (might be smaller for the last batch)
                actual_batch_size = min(batch_size, num_samples - batch_start)
                
                # Generate noise
                z = torch.randn(actual_batch_size, latent_dim).to(device)
                
                # Generate labels
                labels = torch.full((actual_batch_size,), c, dtype=torch.long).to(device)
                
                # Generate images
                gen_imgs = generator(z, labels)
                
                # Normalize to [0, 1]
                gen_imgs = (gen_imgs + 1) / 2
                
                # Save each image in the batch
                for j in range(actual_batch_size):
                    # Convert to PIL image
                    img = transforms.ToPILImage()(gen_imgs[j].cpu())
                    
                    # Save image
                    img_idx = batch_start + j
                    img.save(os.path.join(output_dir, f"class_{c}", f"gen_{img_idx:05d}.png"))
    
    generator.train()


def save_targeted_augmentation(generator, device, latent_dim, output_dir, class_mapping=None):
    """
    Generate and save images for targeted data augmentation with specific counts per class
    Args:
        generator: Trained generator model
        device: Device to run model on
        latent_dim: Dimension of latent space
        output_dir: Directory to save images
        class_mapping: Dictionary mapping class names to indices
    """
    # Verify class mapping contains all required classes
    required_classes = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC']
    
    if class_mapping is None:
        raise ValueError("Class mapping must be provided")
    
    # Check if all required classes are in the mapping
    missing_classes = [cls for cls in required_classes if cls not in class_mapping]
    if missing_classes:
        raise ValueError(f"Missing required classes in mapping: {missing_classes}")
    
    # Print the actual class mapping being used
    print("\nUsing class mapping:")
    for class_name, idx in class_mapping.items():
        print(f"  {class_name}: {idx}")
    
    # Specific augmentation counts as requested by the user
    target_counts = {
        'NV': 0,        # No augmentation for NV (already has 6705 images)
        'MEL': 3340,    # Add 3340 images to reach ~4450 total
        'BKL': 3300,    # Add 3300 images to reach ~4400 total
        'BCC': 2000,    # Add 2000 images to reach ~2500 total
        'AKIEC': 1300   # Add 1300 images to reach ~1600 total
    }
    
    # Convert to index-based counts for save_generated_images
    augmentation_counts = {}
    for class_name, count in target_counts.items():
        if class_name in class_mapping:
            class_idx = class_mapping[class_name]
            augmentation_counts[class_idx] = count
    
    # Create directories for each class
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Generate images for each class with specific counts
    generator.eval()
    
    with torch.no_grad():
        for class_name, count in target_counts.items():
            if count <= 0:
                print(f"Skipping {class_name} (no augmentation needed)")
                continue
                
            class_idx = class_mapping[class_name]
            class_dir = os.path.join(output_dir, class_name)
            
            print(f"Generating {count} images for class {class_name} (index {class_idx})...")
            
            # Generate in batches for efficiency
            batch_size = 16
            for batch_start in range(0, count, batch_size):
                # Calculate actual batch size (might be smaller for the last batch)
                actual_batch_size = min(batch_size, count - batch_start)
                
                # Generate noise
                z = torch.randn(actual_batch_size, latent_dim).to(device)
                
                # Generate labels
                labels = torch.full((actual_batch_size,), class_idx, dtype=torch.long).to(device)
                
                # Generate images
                gen_imgs = generator(z, labels)
                
                # Normalize to [0, 1]
                gen_imgs = (gen_imgs + 1) / 2
                
                # Save each image in the batch
                for j in range(actual_batch_size):
                    # Convert to PIL image
                    img = transforms.ToPILImage()(gen_imgs[j].cpu())
                    
                    # Save image
                    img_idx = batch_start + j
                    img.save(os.path.join(class_dir, f"{class_name}_gen_{img_idx:05d}.png"))
    
    generator.train()
    
    print(f"\nCompleted targeted augmentation:")
    print(f"| Class | Current | Augmented | Total After |")
    print(f"| ----- | ------- | --------- | ----------- |")
    print(f"| NV    | 6705    | +0        | 6705        |")
    print(f"| MEL   | 1113    | +3340     | ~4450       |")
    print(f"| BKL   | 1099    | +3300     | ~4400       |")
    print(f"| BCC   | 514     | +2000     | ~2500       |")
    print(f"| AKIEC | 327     | +1300     | ~1600       |")
    print("\nNote: Generated images are saved in class-specific folders using the class name rather than index.")
    print("      This makes it easier to use them for training later.")

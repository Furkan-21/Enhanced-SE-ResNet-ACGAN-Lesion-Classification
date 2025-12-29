"""
StyleGAN2-ADA utilities for HAM10000 dataset
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pickle
import json
import subprocess
import sys

def setup_stylegan2_ada():
    """
    Set up StyleGAN2-ADA environment
    Returns:
        stylegan_dir: Path to StyleGAN2-ADA directory
    """
    # Check if StyleGAN2-ADA directory exists
    stylegan_dir = os.path.join(os.getcwd(), 'stylegan2-ada-pytorch')
    
    if not os.path.exists(stylegan_dir):
        print("Cloning StyleGAN2-ADA-PyTorch repository...")
        subprocess.run(['git', 'clone', 'https://github.com/NVlabs/stylegan2-ada-pytorch.git'], check=True)
    
    # Add StyleGAN2-ADA directory to path
    if stylegan_dir not in sys.path:
        sys.path.append(stylegan_dir)
    
    return stylegan_dir

def prepare_dataset_for_stylegan(csv_file, img_dir, output_dir, class_name, image_size=128, max_samples=None, resize_method='crop'):
    """
    Prepare dataset for StyleGAN2-ADA training
    Args:
        csv_file: Path to CSV file with image names and labels
        img_dir: Directory with images
        output_dir: Directory to save prepared dataset
        class_name: Class to prepare dataset for
        image_size: Size to resize images to (must be square: 128x128 or 256x256)
        max_samples: Maximum number of samples to include (for debugging)
        resize_method: Method to use for ensuring square images:
                      'crop' - center crop to square
                      'pad' - pad with black to make square
                      'resize' - direct resize (may distort aspect ratio)
    Returns:
        dataset_path: Path to prepared dataset
    """
    # Validate image size is appropriate for StyleGAN2
    if image_size not in [128, 256]:
        print(f"Warning: StyleGAN2 typically works best with image sizes of 128 or 256. Current size: {image_size}")
    
    # Create output directory
    dataset_path = os.path.join(output_dir, f'stylegan2_dataset_{class_name}')
    os.makedirs(dataset_path, exist_ok=True)
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Get class names
    categories = df.columns[1:].tolist()
    
    if class_name not in categories:
        raise ValueError(f"Class {class_name} not found in dataset")
    
    # Filter by class
    filtered_df = df[df[class_name] == 1.0]
    
    # Limit samples if needed
    if max_samples is not None:
        filtered_df = filtered_df.head(max_samples)
    
    print(f"Preparing {len(filtered_df)} images for class {class_name} as {image_size}x{image_size} squares using '{resize_method}' method...")
    
    # Process images
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        img_name = os.path.join(img_dir, row['image'] + '.jpg')
        
        try:
            # Load image
            image = Image.open(img_name).convert('RGB')
            
            # Process image based on chosen method to ensure square format
            if resize_method == 'crop':
                # Center crop to square
                width, height = image.size
                if width != height:
                    new_dim = min(width, height)
                    left = (width - new_dim) // 2
                    top = (height - new_dim) // 2
                    right = left + new_dim
                    bottom = top + new_dim
                    image = image.crop((left, top, right, bottom))
                
                # Resize to target size
                image = image.resize((image_size, image_size), Image.LANCZOS)
                
            elif resize_method == 'pad':
                # Pad to square
                width, height = image.size
                if width != height:
                    new_dim = max(width, height)
                    new_img = Image.new('RGB', (new_dim, new_dim), color='black')
                    paste_x = (new_dim - width) // 2
                    paste_y = (new_dim - height) // 2
                    new_img.paste(image, (paste_x, paste_y))
                    image = new_img
                
                # Resize to target size
                image = image.resize((image_size, image_size), Image.LANCZOS)
                
            else:  # 'resize' or any other value
                # Direct resize (may distort aspect ratio)
                image = image.resize((image_size, image_size), Image.LANCZOS)
            
            # Save image
            output_path = os.path.join(dataset_path, f"{idx:08d}.png")
            image.save(output_path)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    print(f"Dataset prepared at {dataset_path}")
    return dataset_path

def train_stylegan(dataset_path, output_dir, class_name, 
                  image_size=128, batch_size=32, kimg=1000, 
                  resume=None, mirror=True, snap=10):
    """
    Train StyleGAN2-ADA on a dataset
    Args:
        dataset_path: Path to dataset
        output_dir: Directory to save results
        class_name: Class name for naming
        image_size: Image size
        batch_size: Batch size
        kimg: Number of thousands of images to train on
        resume: Path to resume from
        mirror: Whether to use horizontal mirroring
        snap: Snapshot interval
    Returns:
        output_path: Path to trained model
    """
    # Set up StyleGAN2-ADA
    stylegan_dir = setup_stylegan2_ada()
    
    # Create output directory
    output_path = os.path.join(output_dir, f'stylegan2_model_{class_name}')
    os.makedirs(output_path, exist_ok=True)
    
    # Build command
    cmd = [
        'python', os.path.join(stylegan_dir, 'train.py'),
        f'--outdir={output_path}',
        f'--data={dataset_path}',
        f'--cfg=auto',
        f'--res={image_size}',
        f'--batch={batch_size}',
        f'--kimg={kimg}',
        f'--snap={snap}'
    ]
    
    # Add optional arguments
    if resume is not None:
        cmd.append(f'--resume={resume}')
    if mirror:
        cmd.append('--mirror=1')
    else:
        cmd.append('--mirror=0')
    
    # Run training
    print(f"Training StyleGAN2-ADA on {class_name}...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"Training complete. Model saved to {output_path}")
    return output_path

def generate_images(model_path, output_dir, class_name, num_samples, 
                   truncation_psi=0.7, seed=None, noise_mode='const'):
    """
    Generate images using a trained StyleGAN2-ADA model
    Args:
        model_path: Path to trained model directory
        output_dir: Directory to save generated images
        class_name: Class name for naming
        num_samples: Number of images to generate
        truncation_psi: Truncation psi
        seed: Random seed
        noise_mode: Noise mode
    Returns:
        images_dir: Path to generated images
    """
    # Set up StyleGAN2-ADA
    stylegan_dir = setup_stylegan2_ada()
    
    # Find the latest network pickle
    network_pkl = None
    for file in os.listdir(model_path):
        if file.endswith('.pkl'):
            if network_pkl is None or os.path.getmtime(os.path.join(model_path, file)) > os.path.getmtime(network_pkl):
                network_pkl = os.path.join(model_path, file)
    
    if network_pkl is None:
        raise ValueError(f"No .pkl file found in {model_path}")
    
    # Create output directory
    images_dir = os.path.join(output_dir, f'generated_{class_name}')
    os.makedirs(images_dir, exist_ok=True)
    
    # Set seed
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    
    # Build command
    cmd = [
        'python', os.path.join(stylegan_dir, 'generate.py'),
        f'--outdir={images_dir}',
        f'--network={network_pkl}',
        f'--seeds=0-{num_samples-1}',
        f'--trunc={truncation_psi}',
        f'--noise-mode={noise_mode}'
    ]
    
    # Run generation
    print(f"Generating {num_samples} images for class {class_name}...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"Generation complete. Images saved to {images_dir}")
    return images_dir

def calculate_fid_for_stylegan(real_dataset_path, generated_images_path, stylegan_dir=None):
    """
    Calculate FID score for generated images
    Args:
        real_dataset_path: Path to real dataset
        generated_images_path: Path to generated images
        stylegan_dir: Path to StyleGAN2-ADA directory
    Returns:
        fid_score: FID score
    """
    if stylegan_dir is None:
        stylegan_dir = setup_stylegan2_ada()
    
    # Build command
    cmd = [
        'python', os.path.join(stylegan_dir, 'calc_metrics.py'),
        f'--metrics=fid50k_full',
        f'--data={real_dataset_path}',
        f'--network=pickle:{os.path.join(generated_images_path, "network-snapshot-latest.pkl")}'
    ]
    
    # Run FID calculation
    print(f"Calculating FID score...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    # Parse FID score from output
    fid_score = None
    for line in result.stdout.split('\n'):
        if 'fid50k_full' in line:
            fid_score = float(line.split(':')[1].strip())
    
    if fid_score is None:
        print("Warning: Could not parse FID score from output")
        return None
    
    print(f"FID score: {fid_score}")
    return fid_score

def generate_specific_class_counts(model_paths, output_dir, class_counts, truncation_psi=0.7):
    """
    Generate specific numbers of images for each class
    Args:
        model_paths: Dictionary mapping class names to model paths
        output_dir: Directory to save generated images
        class_counts: Dictionary mapping class names to number of images to generate
        truncation_psi: Truncation psi
    Returns:
        output_dirs: Dictionary mapping class names to output directories
    """
    output_dirs = {}
    
    for class_name, count in class_counts.items():
        if class_name not in model_paths:
            print(f"Warning: No model found for class {class_name}")
            continue
        
        print(f"Generating {count} images for class {class_name}...")
        output_dirs[class_name] = generate_images(
            model_paths[class_name], 
            output_dir, 
            class_name, 
            count, 
            truncation_psi=truncation_psi
        )
    
    return output_dirs

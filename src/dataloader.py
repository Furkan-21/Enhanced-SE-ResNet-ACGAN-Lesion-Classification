import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# === Use hair-removed images directory by default ===
HAIR_REMOVED_IMG_DIR = 'archive/hair_removed_images'  # Change this to your processed folder

class HAM10000Dataset(Dataset):
    """
    Dataset class for the HAM10000 skin lesion classification dataset
    """
    def __init__(self, csv_file, img_dir=HAIR_REMOVED_IMG_DIR, mask_dir=None, transform=None, use_masks=False):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            mask_dir (string, optional): Directory with all the segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_masks (bool): Whether to use segmentation masks or not.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.use_masks = use_masks
        
        # Convert categories to indices for classification
        self.categories = self.data_frame.columns[1:].tolist()
        
        # Extract image names and corresponding labels
        self.image_filenames = self.data_frame['image'].values
        
        # For multi-class classification, convert one-hot encoded labels to class indices
        # Each row has a 1.0 in the column corresponding to the correct class
        self.labels = np.argmax(self.data_frame[self.categories].values, axis=1)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_name = os.path.join(self.img_dir, self.image_filenames[idx] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        
        # Load mask if required
        if self.use_masks and self.mask_dir is not None:
            mask_name = os.path.join(self.mask_dir, self.image_filenames[idx] + '_segmentation.jpg')
            mask = Image.open(mask_name).convert('L')
            
            # Apply transformations to both image and mask if provided
            if self.transform:
                image = self.transform(image)
                mask = transforms.ToTensor()(mask)
            
            # Get label (now a single integer for the class)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return {'image': image, 'mask': mask, 'label': label, 'image_name': self.image_filenames[idx]}
        else:
            # Apply transformations to image if provided
            if self.transform:
                image = self.transform(image)
            
            # Get label (now a single integer for the class)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return {'image': image, 'label': label, 'image_name': self.image_filenames[idx]}

def get_data_loaders(csv_file, img_dir=HAIR_REMOVED_IMG_DIR, mask_dir=None, batch_size=32, use_masks=False, val_split=0.1, test_split=0.1):
    """
    Create data loaders for training, validation and testing
    
    Args:
        csv_file (string): Path to the CSV file with annotations
        img_dir (string): Directory with all the images (default: hair-removed)
        mask_dir (string, optional): Directory with all the segmentation masks
        batch_size (int): Batch size for the data loaders
        use_masks (bool): Whether to use segmentation masks or not
        val_split (float): Fraction of data to use for validation
        test_split (float): Fraction of data to use for testing
        
    Returns:
        dict: Dictionary containing train, val, and test data loaders
    """
    # Define transformations for training, validation, and testing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read the dataset
    df = pd.read_csv(csv_file)
    
    # Create indices for train/val/test split
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    test_size = int(len(df) * test_split)
    val_size = int(len(df) * val_split)
    train_size = len(df) - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create DataFrames for each split
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    # Save split dataframes to the data directory relative to this script
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_dir, exist_ok=True) 
    
    train_csv_path = os.path.join(data_dir, 'train_split.csv')
    val_csv_path = os.path.join(data_dir, 'val_split.csv')
    test_csv_path = os.path.join(data_dir, 'test_split.csv')
    
    # Save split dataframes if needed for reproducibility
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    # Create datasets
    train_dataset = HAM10000Dataset(
        csv_file=train_csv_path, 
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=train_transform,
        use_masks=use_masks
    )
    
    val_dataset = HAM10000Dataset(
        csv_file=val_csv_path, 
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=val_test_transform,
        use_masks=use_masks
    )
    
    test_dataset = HAM10000Dataset(
        csv_file=test_csv_path, 
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=val_test_transform,
        use_masks=use_masks
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'categories': train_dataset.categories
    }
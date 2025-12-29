# src/utils.py
"""
Utility functions for skin lesion classification project.
"""
import cv2
import os
import numpy as np

def remove_hairs_from_image(image: np.ndarray) -> np.ndarray:
    """
    Remove hairs from a dermoscopic image using morphological operations and inpainting.
    Args:
        image (np.ndarray): Input image (BGR or RGB).
    Returns:
        np.ndarray: Image with hairs removed (same shape as input).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image
    # Kernel for morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # Blackhat to find hair contours
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Threshold to create mask
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint to remove hair
    inpainted = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return inpainted

def batch_remove_hairs(source_dir: str, target_dir: str, extensions=(".jpg", ".png")):
    """
    Remove hairs from all images in a directory and save results to another directory.
    Args:
        source_dir (str): Directory with input images.
        target_dir (str): Directory to save processed images.
        extensions (tuple): Allowed image extensions.
    """
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(extensions):
            img_path = os.path.join(source_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
            # Convert BGR to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = remove_hairs_from_image(image_rgb)
            # Convert back to BGR for saving
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(target_dir, filename)
            cv2.imwrite(save_path, result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print("Hair removing complete.")

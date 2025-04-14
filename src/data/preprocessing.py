import os
import cv2
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_images(image_dir, output_dir=None, size=(640, 640), augment=False):
    """
    Preprocess images for training
    
    Args:
        image_dir (str): Directory containing images to preprocess
        output_dir (str, optional): Directory to save preprocessed images. If None, images are processed in-place.
        size (tuple): Target size (width, height) for preprocessing
        augment (bool): Whether to apply data augmentation
        
    Returns:
        list: Paths to preprocessed images
    """
    logger.info(f"Preprocessing images in {image_dir}...")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [
        f for f in os.listdir(image_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    processed_images = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        try:
            # Read image
            img = cv2.imread(img_path)
            
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            
            # Resize image
            img_resized = cv2.resize(img, size)
            
            # Save processed image
            if output_dir:
                save_path = os.path.join(output_dir, img_file)
                cv2.imwrite(save_path, img_resized)
                processed_images.append(save_path)
            else:
                # Process in-place
                cv2.imwrite(img_path, img_resized)
                processed_images.append(img_path)
            
            # Perform augmentation if requested
            if augment and output_dir:
                # Horizontal flip
                img_flipped = cv2.flip(img_resized, 1)
                flip_path = os.path.join(output_dir, f"flip_{img_file}")
                cv2.imwrite(flip_path, img_flipped)
                processed_images.append(flip_path)
                
                # Brightness adjustment
                img_bright = cv2.convertScaleAbs(img_resized, alpha=1.2, beta=10)
                bright_path = os.path.join(output_dir, f"bright_{img_file}")
                cv2.imwrite(bright_path, img_bright)
                processed_images.append(bright_path)
                
                # Contrast adjustment
                img_contrast = cv2.convertScaleAbs(img_resized, alpha=1.3, beta=0)
                contrast_path = os.path.join(output_dir, f"contrast_{img_file}")
                cv2.imwrite(contrast_path, img_contrast)
                processed_images.append(contrast_path)
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
    
    logger.info(f"Processed {len(processed_images)} images")
    return processed_images 
import os
import json
import shutil
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_dataset(dataset_path='datasets'):
    """
    Verify the dataset structure and annotations
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    logger.info("Verifying dataset structure...")

    # Check for required folders
    required_folders = ['train', 'valid', 'test']
    for folder in required_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            logger.warning(f"Warning: {folder} folder not found in {dataset_path}")
            return False

        # Check for images and labels folders
        images_path = os.path.join(folder_path, 'images')
        labels_path = os.path.join(folder_path, 'labels')

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            logger.warning(f"Warning: {folder} folder must contain 'images' and 'labels' subfolders")
            return False

    # Count files
    stats = {}
    for folder in required_folders:
        images_path = os.path.join(dataset_path, folder, 'images')
        labels_path = os.path.join(dataset_path, folder, 'labels')

        images_count = len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        labels_count = len([f for f in os.listdir(labels_path) if f.lower().endswith('.txt')])

        stats[folder] = {
            'images': images_count,
            'labels': labels_count
        }

    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    
    return True

def prepare_dataset(dataset_path='datasets'):
    """
    Prepare dataset for training by creating data.yaml
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        str: Path to the data.yaml file
    """
    logger.info(f"Preparing dataset from {dataset_path}...")
    
    # Get absolute paths
    abs_path = os.path.abspath(dataset_path)
    
    # Create data.yaml configuration
    data_yaml = {
        'train': os.path.join(abs_path, 'train'),
        'val': os.path.join(abs_path, 'valid'),
        'test': os.path.join(abs_path, 'test'),
        'nc': 1,  # Number of classes
        'names': ['Spill']  # Class names
    }
    
    # Write data.yaml to config directory
    config_path = os.path.join('config', 'data.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    logger.info(f"Created data configuration: {config_path}")
    return config_path 
#!/usr/bin/env python3
"""
Spill Detection - Lightweight Model Training Script

This script creates a lightweight model from a primary model through
knowledge distillation. The lightweight model is optimized for smaller size
while maintaining acceptable performance.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import prepare_dataset, verify_dataset
from src.models.lightweight_model import LightweightModel
from src.utils.visualization import visualize_predictions
from src.utils.metrics import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lightweight_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_lightweight_model(data_dir="datasets", output_dir="models", 
                           primary_model_path=None, epochs=20, 
                           batch_size=16, image_size=416, visualize=True):
    """
    Train the lightweight spill detection model with optional distillation
    
    Args:
        data_dir (str): Path to dataset directory
        output_dir (str): Directory to save trained model
        primary_model_path (str, optional): Path to primary model for distillation
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        image_size (int): Image size for training
        visualize (bool): Whether to visualize results
        
    Returns:
        str: Path to the trained model
    """
    logger.info("=== Spill Detection Lightweight Model Training ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    if primary_model_path:
        logger.info(f"Primary model for distillation: {primary_model_path}")
        if not os.path.exists(primary_model_path):
            logger.warning(f"Primary model not found at {primary_model_path}. Training from scratch.")
            primary_model_path = None
    else:
        logger.info("Training lightweight model from scratch (no distillation)")
    
    logger.info(f"Training configuration: {epochs} epochs, batch size {batch_size}, image size {image_size}")
    
    # Verify dataset
    if not verify_dataset(data_dir):
        logger.error("Dataset verification failed. Please check your dataset structure.")
        return None
    
    # Prepare dataset configuration
    data_yaml = prepare_dataset(data_dir)
    
    # Initialize lightweight model
    lightweight_model = LightweightModel()
    
    # Determine whether to use distillation or train from scratch
    if primary_model_path and os.path.exists(primary_model_path):
        logger.info("Creating lightweight model through distillation")
        model_path = lightweight_model.create_from_primary(
            primary_model_path=primary_model_path,
            data_yaml=data_yaml,
            output_dir=output_dir
        )
    else:
        logger.info("Training lightweight model from scratch")
        lightweight_model.initialize_model()
        model_path = lightweight_model.train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            output_dir=output_dir
        )
    
    if model_path and os.path.exists(model_path):
        # Check model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Lightweight model created successfully and saved to: {model_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        # Check if model meets size requirements
        if model_size_mb > 10:
            logger.warning(f"WARNING: Model exceeds 10MB size limit ({model_size_mb:.2f} MB)")
        else:
            logger.info(f"Model meets the size requirement (less than 10MB)")
        
        # Evaluate model
        metrics_dict = lightweight_model.evaluate(model_path, data_yaml)
        
        # Calculate and format metrics
        metrics = calculate_metrics(
            metrics_dict,
            save_path=os.path.join('results', 'lightweight_metrics.json')
        )
        
        # Visualize predictions if enabled
        if visualize:
            test_images_dir = os.path.join(data_dir, 'test', 'images')
            if os.path.exists(test_images_dir):
                visualize_predictions(
                    model_path,
                    test_images_dir,
                    output_dir='results/lightweight_model_predictions'
                )
                logger.info(f"Prediction visualizations saved to results/lightweight_model_predictions")
        
        return model_path
    else:
        logger.error("Failed to create lightweight model.")
        return None

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Create lightweight spill detection model")
    
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Path to dataset directory (default: datasets)")
    
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save trained model (default: models)")
    
    parser.add_argument("--primary_model", type=str, default="models/primary_spill_detection.pt",
                        help="Path to primary model for distillation (default: models/primary_spill_detection.pt)")
    
    parser.add_argument("--no_distillation", action="store_true",
                        help="Train lightweight model from scratch without using the primary model")
    
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size (default: 16)")
    
    parser.add_argument("--image_size", type=int, default=416,
                        help="Image size for training (default: 416)")
    
    parser.add_argument("--no_visualize", action="store_true",
                        help="Disable result visualization")
    
    return parser.parse_args()

def main():
    """Main function to run the lightweight model creation"""
    # Parse command-line arguments
    args = parse_args()
    
    # Determine whether to use primary model for distillation
    primary_model_path = None if args.no_distillation else args.primary_model
    
    # Train the lightweight model
    train_lightweight_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        primary_model_path=primary_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        visualize=not args.no_visualize
    )

if __name__ == "__main__":
    main() 
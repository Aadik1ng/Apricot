#!/usr/bin/env python3
"""
Spill Detection - Primary Model Training Script

This script trains the primary spill detection model using the YOLOv8 architecture.
The model is designed to detect liquid spills in images, with size up to 75MB.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import prepare_dataset, verify_dataset
from src.models.primary_model import PrimaryModel
from src.utils.visualization import visualize_predictions
from src.utils.metrics import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("primary_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_primary_model(data_dir="datasets", output_dir="models", epochs=100, 
                        batch_size=8, image_size=640, visualize=True):
    """
    Train the primary spill detection model
    
    Args:
        data_dir (str): Path to dataset directory
        output_dir (str): Directory to save trained model
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        image_size (int): Image size for training
        visualize (bool): Whether to visualize results
        
    Returns:
        str: Path to the trained model
    """
    logger.info("=== Spill Detection Primary Model Training ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training configuration: {epochs} epochs, batch size {batch_size}, image size {image_size}")
    
    # Verify dataset
    if not verify_dataset(data_dir):
        logger.error("Dataset verification failed. Please check your dataset structure.")
        return None
    
    # Prepare dataset configuration
    data_yaml = prepare_dataset(data_dir)
    
    # Initialize primary model
    primary_model = PrimaryModel()
    primary_model.initialize_model()
    
    # Train model
    model_path = primary_model.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        output_dir=output_dir
    )
    
    if model_path and os.path.exists(model_path):
        # Check model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Training complete. Model saved to: {model_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        # Check if model meets size requirements
        if model_size_mb > 75:
            logger.warning(f"WARNING: Model exceeds 75MB size limit ({model_size_mb:.2f} MB)")
        
        # Evaluate model
        metrics_dict = primary_model.evaluate(model_path, data_yaml)
        
        # Calculate and format metrics
        metrics = calculate_metrics(
            metrics_dict, 
            save_path=os.path.join('results', 'primary_metrics.json')
        )
        
        # Visualize predictions if enabled
        if visualize:
            test_images_dir = os.path.join(data_dir, 'test', 'images')
            if os.path.exists(test_images_dir):
                visualize_predictions(
                    model_path,
                    test_images_dir,
                    output_dir='results/primary_model_predictions'
                )
                logger.info(f"Prediction visualizations saved to results/primary_model_predictions")
        
        return model_path
    else:
        logger.error("Training failed to produce a model.")
        return None

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train primary spill detection model")
    
    parser.add_argument("--data_dir", type=str, default="datasets",
                        help="Path to dataset directory (default: datasets)")
    
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save trained model (default: models)")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size (default: 8)")
    
    parser.add_argument("--image_size", type=int, default=640,
                        help="Image size for training (default: 640)")
    
    parser.add_argument("--no_visualize", action="store_true",
                        help="Disable result visualization")
    
    return parser.parse_args()

def main():
    """Main function to run the primary model training"""
    # Parse command-line arguments
    args = parse_args()
    
    # Train the model
    train_primary_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        visualize=not args.no_visualize
    )

if __name__ == "__main__":
    main() 
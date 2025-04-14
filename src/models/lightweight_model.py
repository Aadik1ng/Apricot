import os
import torch
import logging
import shutil
from ultralytics import YOLO
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightweightModel:
    def __init__(self, config_path='config/model_config.yaml'):
        """
        Initialize the lightweight spill detection model
        
        Args:
            config_path (str): Path to model configuration file
        """
        self.model = None
        self.config_path = config_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def initialize_model(self, model_type='yolov8n.pt'):
        """
        Initialize the YOLO model
        
        Args:
            model_type (str): Type of YOLOv8 model to use (yolov8n is the nano version)
            
        Returns:
            YOLO: Initialized YOLO model
        """
        logger.info(f"Initializing lightweight model using {model_type}")
        self.model = YOLO(model_type)
        return self.model
    
    def train(self, data_yaml, epochs=20, batch_size=16, image_size=416, output_dir='models'):
        """
        Train the lightweight detection model
        
        Args:
            data_yaml (str): Path to data configuration file
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            image_size (int): Input image size (reduced for lightweight model)
            output_dir (str): Directory to save the trained model
            
        Returns:
            str: Path to the trained model
        """
        logger.info("Starting training for lightweight model")
        
        if self.model is None:
            self.initialize_model()
        
        # Define hyperparameters for lightweight model (optimized for speed and size)
        hyperparams = {
            # Learning parameters
            'lr0': 0.001,          # Initial learning rate
            'lrf': 0.01,           # Final learning rate factor
            'momentum': 0.937,     # SGD momentum
            'weight_decay': 0.0005,# Weight decay

            # Simpler augmentation for faster training
            'mosaic': 0.5,         # Reduced mosaic augmentation
            'mixup': 0.0,          # No mixup for faster training
            'copy_paste': 0.0,     # No copy-paste for faster training

            # Regularization
            'label_smoothing': 0.0,# No label smoothing for simpler model

            # Simplified image transformations
            'degrees': 5.0,        # Reduced rotation range
            'translate': 0.1,      # Translation factor
            'scale': 0.3,          # Reduced scaling factor
            'shear': 0.0,          # No shear for simpler augmentation
            'perspective': 0.0,    # No perspective transform

            # Only horizontal flip for simplicity
            'flipud': 0.0,         # No vertical flip
            'fliplr': 0.5,         # Horizontal flip

            # Reduced color jittering
            'hsv_h': 0.01,         # Minimal hue variation
            'hsv_s': 0.3,          # Reduced saturation variation
            'hsv_v': 0.2,          # Reduced brightness variation
        }
        
        # Start training
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            patience=10,
            device=self.device,
            workers=2,
            optimizer='Adam',
            plots=True,
            verbose=True,
            project='runs',
            name='lightweight_model',
            **hyperparams
        )
        
        # Save the model
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join('runs', 'lightweight_model', 'weights', 'best.pt')
        save_path = os.path.join(output_dir, 'lightweight_spill_detection.pt')
        
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, save_path)
            logger.info(f"Best lightweight model saved to {save_path}")
            
            # Log model size
            model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            logger.info(f"Lightweight model size: {model_size_mb:.2f} MB")
            
            # Check if model meets size requirement
            if model_size_mb > 10:
                logger.warning(f"Lightweight model exceeds 10MB limit: {model_size_mb:.2f}MB")
            
            return save_path
        else:
            logger.error(f"Training did not produce a model at {best_model_path}")
            return None
    
    def create_from_primary(self, primary_model_path, data_yaml, output_dir='models'):
        """
        Create a lightweight model from a primary model through knowledge distillation
        
        Args:
            primary_model_path (str): Path to the primary model
            data_yaml (str): Path to data configuration file
            output_dir (str): Directory to save the lightweight model
            
        Returns:
            str: Path to the lightweight model
        """
        logger.info(f"Creating lightweight model from primary model: {primary_model_path}")
        
        # First initialize a nano model
        self.initialize_model('yolov8n.pt')
        
        # Train with fewer epochs since we're refining rather than starting from scratch
        save_path = self.train(
            data_yaml=data_yaml,
            epochs=20,
            batch_size=16,
            image_size=416,
            output_dir=output_dir
        )
        
        if save_path and os.path.exists(save_path):
            # Print model size comparison
            primary_size = os.path.getsize(primary_model_path) / (1024 * 1024)  # MB
            light_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            
            logger.info(f"Original model size: {primary_size:.2f} MB")
            logger.info(f"Lightweight model size: {light_size:.2f} MB")
            logger.info(f"Size reduction: {((primary_size - light_size) / primary_size * 100):.1f}%")
            
        return save_path
    
    def evaluate(self, model_path, data_yaml):
        """
        Evaluate the lightweight model
        
        Args:
            model_path (str): Path to the model
            data_yaml (str): Path to data configuration file
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating lightweight model: {model_path}")
        
        # Load the model
        eval_model = YOLO(model_path)
        
        # Run validation
        metrics = eval_model.val(
            data=data_yaml,
            plots=True,
            conf=0.25,
            iou=0.45
        )
        
        # Log metrics
        logger.info("Lightweight Model Metrics:")
        logger.info(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        logger.info(f"Recall: {metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        logger.info(f"mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        return metrics.results_dict 
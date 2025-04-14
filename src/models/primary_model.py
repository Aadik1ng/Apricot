import os
import torch
import logging
from ultralytics import YOLO
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PrimaryModel:
    def __init__(self, config_path='config/model_config.yaml'):
        """
        Initialize the primary spill detection model
        
        Args:
            config_path (str): Path to model configuration file
        """
        self.model = None
        self.config_path = config_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def initialize_model(self, model_type='yolov8m.pt'):
        """
        Initialize the YOLO model
        
        Args:
            model_type (str): Type of YOLOv8 model to use
            
        Returns:
            YOLO: Initialized YOLO model
        """
        logger.info(f"Initializing primary model using {model_type}")
        self.model = YOLO(model_type)
        return self.model
    
    def train(self, data_yaml, epochs=100, batch_size=8, image_size=640, output_dir='models'):
        """
        Train the primary detection model
        
        Args:
            data_yaml (str): Path to data configuration file
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            image_size (int): Input image size
            output_dir (str): Directory to save the trained model
            
        Returns:
            str: Path to the trained model
        """
        logger.info("Starting training for primary model")
        
        if self.model is None:
            self.initialize_model()
        
        # Define hyperparameters
        hyperparams = {
            # Learning parameters
            'lr0': 0.01,           # Initial learning rate
            'lrf': 0.01,           # Final learning rate factor
            'momentum': 0.937,     # SGD momentum
            'weight_decay': 0.0005,# Weight decay

            # Augmentation strategies
            'mosaic': 1.0,         # Mosaic augmentation
            'mixup': 0.2,          # Image mixup probability
            'copy_paste': 0.1,     # Copy-paste augmentation

            # Regularization
            'label_smoothing': 0.1,# Label smoothing

            # Loss configuration
            'box': 7.5,            # Bounding box loss gain
            'cls': 0.5,            # Classification loss gain
            'dfl': 1.5,            # Distribution focal loss

            # Image transformations
            'degrees': 10.0,       # Rotation range
            'translate': 0.1,      # Translation factor
            'scale': 0.5,          # Scaling factor
            'shear': 5.0,          # Shear angle
            'perspective': 0.0,    # Perspective transform

            # Flip augmentations
            'flipud': 0.0,         # Vertical flip
            'fliplr': 0.5,         # Horizontal flip

            # Color jittering
            'hsv_h': 0.015,        # Hue variation
            'hsv_s': 0.7,          # Saturation variation
            'hsv_v': 0.4,          # Value variation
        }
        
        # Start training
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            patience=20,
            device=self.device,
            workers=2,
            optimizer='Adam',
            plots=True,
            verbose=True,
            project='runs',
            name='primary_model',
            **hyperparams
        )
        
        # Save the model
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join('runs', 'primary_model', 'weights', 'best.pt')
        save_path = os.path.join(output_dir, 'primary_spill_detection.pt')
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, save_path)
            logger.info(f"Best model saved to {save_path}")
            
            # Log model size
            model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            logger.info(f"Primary model size: {model_size_mb:.2f} MB")
            
            return save_path
        else:
            logger.error(f"Training did not produce a model at {best_model_path}")
            return None
    
    def evaluate(self, model_path, data_yaml):
        """
        Evaluate the trained model
        
        Args:
            model_path (str): Path to the trained model
            data_yaml (str): Path to data configuration file
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating model: {model_path}")
        
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
        logger.info("Evaluation metrics:")
        logger.info(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        logger.info(f"Recall: {metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        logger.info(f"mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        return metrics.results_dict 
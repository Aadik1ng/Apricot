import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from ultralytics import YOLO
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def visualize_predictions(model_path, test_dir, output_dir='results', conf_threshold=0.25):
    """
    Visualize model predictions on test images
    
    Args:
        model_path (str): Path to the trained model
        test_dir (str): Directory containing test images
        output_dir (str): Directory to save visualizations
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        list: Paths to visualization images
    """
    logger.info(f"Visualizing predictions from model: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Get all image files
    image_files = [
        f for f in os.listdir(test_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    # Sort files for consistent visualization
    image_files.sort()
    
    visualization_paths = []
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        
        # Run inference
        results = model(img_path, conf=conf_threshold)
        
        # Process results for visualization
        for i, result in enumerate(results):
            # Plot results with bounding boxes
            plot = result.plot()
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"vis_{img_file}")
            cv2.imwrite(vis_path, plot)
            visualization_paths.append(vis_path)
            
            # Log detection info
            boxes = result.boxes
            logger.info(f"Image: {img_file}, Detections: {len(boxes)}")
            
            # Print confidence scores
            if len(boxes) > 0:
                conf_scores = boxes.conf.tolist()
                logger.info(f"  Confidence scores: {[f'{score:.2f}' for score in conf_scores]}")
    
    logger.info(f"Created {len(visualization_paths)} visualization images in {output_dir}")
    return visualization_paths

def plot_metrics(metrics, output_dir='results', name='metrics'):
    """
    Plot model performance metrics
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        output_dir (str): Directory to save plots
        name (str): Name prefix for saved plots
        
    Returns:
        list: Paths to saved plots
    """
    logger.info(f"Plotting metrics: {name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = []
    
    # Get key metrics
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    
    # Plot precision-recall bar chart
    plt.figure(figsize=(10, 6))
    metrics_names = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
    metrics_values = [precision, recall, map50, map50_95]
    
    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1.0)
    plt.title('Detection Performance Metrics')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Save the plot
    metrics_path = os.path.join(output_dir, f"{name}_performance.png")
    plt.savefig(metrics_path)
    plot_paths.append(metrics_path)
    plt.close()
    
    # Create confusion matrix if available
    confusion_matrix = metrics.get('confusion_matrix', None)
    if confusion_matrix is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        classes = ['Background', 'Spill']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, f"{name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plot_paths.append(cm_path)
        plt.close()
    
    logger.info(f"Saved metric plots to {output_dir}")
    return plot_paths 
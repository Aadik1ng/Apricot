import logging
import numpy as np
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_metrics(results_dict, save_path=None):
    """
    Calculate and format model evaluation metrics
    
    Args:
        results_dict (dict): Results dictionary from YOLO model evaluation
        save_path (str, optional): Path to save metrics JSON
        
    Returns:
        dict: Formatted metrics dictionary
    """
    logger.info("Calculating performance metrics")
    
    # Extract key metrics
    metrics = {
        'precision': results_dict.get('metrics/precision(B)', 0),
        'recall': results_dict.get('metrics/recall(B)', 0),
        'mAP50': results_dict.get('metrics/mAP50(B)', 0),
        'mAP50-95': results_dict.get('metrics/mAP50-95(B)', 0),
        'f1_score': 0
    }
    
    # Calculate F1 score if precision and recall are available
    if metrics['precision'] > 0 or metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-16)
    
    # Format metrics for display
    formatted_metrics = {
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'mAP50': f"{metrics['mAP50']:.4f}",
        'mAP50-95': f"{metrics['mAP50-95']:.4f}",
        'F1 Score': f"{metrics['f1_score']:.4f}"
    }
    
    # Log metrics
    logger.info("Evaluation Metrics:")
    for metric_name, metric_value in formatted_metrics.items():
        logger.info(f"  {metric_name}: {metric_value}")
    
    # Save metrics to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(formatted_metrics, f, indent=2)
        logger.info(f"Metrics saved to {save_path}")
    
    return formatted_metrics

def compare_models(primary_metrics, lightweight_metrics, save_path=None):
    """
    Compare metrics between primary and lightweight models
    
    Args:
        primary_metrics (dict): Metrics from primary model
        lightweight_metrics (dict): Metrics from lightweight model
        save_path (str, optional): Path to save comparison JSON
        
    Returns:
        dict: Model comparison metrics
    """
    logger.info("Comparing model performance")
    
    comparison = {}
    
    # Calculate differences
    for metric in ['Precision', 'Recall', 'mAP50', 'mAP50-95', 'F1 Score']:
        primary_value = float(primary_metrics.get(metric, 0))
        lightweight_value = float(lightweight_metrics.get(metric, 0))
        
        abs_diff = primary_value - lightweight_value
        rel_diff = abs_diff / (primary_value + 1e-16) * 100  # Percentage change
        
        comparison[metric] = {
            'primary': primary_value,
            'lightweight': lightweight_value,
            'absolute_difference': abs_diff,
            'relative_difference_percent': rel_diff
        }
    
    # Log comparison
    logger.info("Model Comparison:")
    for metric, values in comparison.items():
        logger.info(f"  {metric}:")
        logger.info(f"    Primary: {values['primary']:.4f}")
        logger.info(f"    Lightweight: {values['lightweight']:.4f}")
        logger.info(f"    Difference: {values['absolute_difference']:.4f} ({values['relative_difference_percent']:.2f}%)")
    
    # Save comparison to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            # Convert to serializable format
            serializable_comparison = {}
            for metric, values in comparison.items():
                serializable_comparison[metric] = {
                    'primary': float(values['primary']),
                    'lightweight': float(values['lightweight']),
                    'absolute_difference': float(values['absolute_difference']),
                    'relative_difference_percent': float(values['relative_difference_percent'])
                }
            json.dump(serializable_comparison, f, indent=2)
        logger.info(f"Comparison saved to {save_path}")
    
    return comparison 
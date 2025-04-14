#!/usr/bin/env python3
"""
Comprehensive Spill Detection Testing Script

This script provides a unified testing approach for spill detection models, including:
1. Unit tests for model structure and properties
2. Performance tests on real images
3. Comparison between primary and lightweight models

Run with: python -m tests.test_spill_detection [--flags]
"""

import os
import sys
import unittest
import argparse
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.primary_model import PrimaryModel
from src.models.lightweight_model import LightweightModel
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spill_detection_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PRIMARY_MODEL_PATH = "models/primary_spill_detection.pt"
LIGHTWEIGHT_MODEL_PATH = "models/lightweight_spill_detection.pt"
DEFAULT_TEST_DIR = "datasets/test/images"
DEFAULT_OUTPUT_DIR = "results/model_tests"

###################
# Unit Test Classes
###################

class TestPrimaryModel(unittest.TestCase):
    """Test cases for the primary model"""
    
    def setUp(self):
        """Setup the test environment"""
        self.model = PrimaryModel()
        self.model_path = PRIMARY_MODEL_PATH
    
    def test_initialization(self):
        """Test model initialization"""
        # Test model is initialized correctly
        self.assertIsNone(self.model.model)
        
        # Test model initialization
        initialized_model = self.model.initialize_model()
        self.assertIsNotNone(initialized_model)
    
    def test_model_properties(self):
        """Test model properties"""
        # Initialize model for testing
        self.model.initialize_model()
        
        # Check model task attribute
        self.assertEqual(self.model.model.task, 'detect')
        
        # Check that model has the necessary attributes
        self.assertTrue(hasattr(self.model.model, 'model'))
        self.assertTrue(hasattr(self.model.model, 'ckpt_path'))
    
    def test_model_size_requirements(self):
        """Test model meets size requirements"""
        if os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            logger.info(f"Primary model size: {size_mb:.2f} MB")
            self.assertLessEqual(size_mb, 75, f"Primary model exceeds 75MB size limit: {size_mb:.2f}MB")
        else:
            logger.warning(f"Primary model not found at {self.model_path}, skipping size test")
            self.skipTest(f"Model file not found: {self.model_path}")


class TestLightweightModel(unittest.TestCase):
    """Test cases for the lightweight model"""
    
    def setUp(self):
        """Setup the test environment"""
        self.model = LightweightModel()
        self.model_path = LIGHTWEIGHT_MODEL_PATH
    
    def test_initialization(self):
        """Test model initialization"""
        # Test model is initialized correctly
        self.assertIsNone(self.model.model)
        
        # Test model initialization
        initialized_model = self.model.initialize_model()
        self.assertIsNotNone(initialized_model)
    
    def test_model_properties(self):
        """Test model properties"""
        # Initialize model for testing
        self.model.initialize_model()
        
        # Check model task attribute
        self.assertEqual(self.model.model.task, 'detect')
        
        # Check that model has the necessary attributes
        self.assertTrue(hasattr(self.model.model, 'model'))
        self.assertTrue(hasattr(self.model.model, 'ckpt_path'))
    
    def test_model_size_requirements(self):
        """Test model meets size requirements"""
        if os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            logger.info(f"Lightweight model size: {size_mb:.2f} MB")
            self.assertLessEqual(size_mb, 10, f"Lightweight model exceeds 10MB size limit: {size_mb:.2f}MB")
        else:
            logger.warning(f"Lightweight model not found at {self.model_path}, skipping size test")
            self.skipTest(f"Model file not found: {self.model_path}")


########################
# Image Testing Classes
########################

class ModelTester:
    def __init__(self, model_path):
        """
        Initialize model tester
        
        Args:
            model_path (str): Path to trained YOLO model
        """
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        
        # Determine model type based on size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self.model_type = "lightweight" if model_size_mb < 15 else "primary"
        logger.info(f"Model loaded successfully (type: {self.model_type}, size: {model_size_mb:.2f} MB)")
    
    def test_on_images(self, image_dir, output_dir='results', conf_threshold=0.25):
        """
        Test model on a directory of images
        
        Args:
            image_dir (str): Directory containing test images
            output_dir (str): Directory to save result images
            conf_threshold (float): Confidence threshold for predictions
            
        Returns:
            list: List of detection results
        """
        logger.info(f"Testing model on images in {image_dir}")
        
        # Create model-specific output directory
        output_dir = os.path.join(output_dir, f"{self.model_type}_model_predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {image_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images")
        
        # Test results storage
        test_results = []
        
        # Process each image
        for img_name in image_files:
            # Full path to image
            img_path = os.path.join(image_dir, img_name)
            
            # Run inference
            results = self.model(img_path, conf=conf_threshold)
            
            # Process results
            for result in results:
                # Plot results
                plot = result.plot()
                
                # Save plot
                output_path = os.path.join(output_dir, f'prediction_{img_name}')
                cv2.imwrite(output_path, plot)
                
                # Collect detection information
                boxes = result.boxes
                test_results.append({
                    'image': img_name,
                    'detections': len(boxes),
                    'confidence': boxes.conf.tolist() if len(boxes) > 0 else []
                })
                
                logger.info(f"Image: {img_name}, Detections: {len(boxes)}")
        
        # Print summary
        self.print_test_summary(test_results)
        
        # Create visualization plots
        self.create_visualization(test_results, output_dir)
        
        return test_results
    
    def print_test_summary(self, test_results):
        """
        Print summary of test results
        
        Args:
            test_results (list): List of detection results
        """
        logger.info("\n--- Model Testing Summary ---")
        logger.info(f"Total Images Tested: {len(test_results)}")
        
        # Aggregate statistics
        total_detections = sum(result['detections'] for result in test_results)
        avg_detections_per_image = total_detections / len(test_results) if test_results else 0
        
        logger.info(f"Total Detections: {total_detections}")
        logger.info(f"Average Detections per Image: {avg_detections_per_image:.2f}")
        
        # Confidence analysis
        all_confidences = [
            conf 
            for result in test_results 
            for conf in result['confidence']
        ]
        
        if all_confidences:
            logger.info("\nConfidence Analysis:")
            logger.info(f"  Mean Confidence: {sum(all_confidences) / len(all_confidences):.4f}")
            logger.info(f"  Max Confidence: {max(all_confidences):.4f}")
            logger.info(f"  Min Confidence: {min(all_confidences):.4f}")
    
    def create_visualization(self, test_results, output_dir):
        """
        Create visualization plots for test results
        
        Args:
            test_results (list): List of detection results
            output_dir (str): Directory to save visualizations
        """
        if not test_results:
            return
        
        # Create a detections-per-image bar chart
        plt.figure(figsize=(12, 6))
        images = [result['image'] for result in test_results]
        detections = [result['detections'] for result in test_results]
        
        plt.bar(images, detections, color='skyblue')
        plt.xlabel('Images')
        plt.ylabel('Number of Detections')
        plt.title(f'Detections per Image - {self.model_type.capitalize()} Model')
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detections_per_image.png'))
        plt.close()
        
        # Create confidence histogram if there are detections
        all_confidences = [
            conf 
            for result in test_results 
            for conf in result['confidence']
        ]
        
        if all_confidences:
            plt.figure(figsize=(10, 6))
            plt.hist(all_confidences, bins=20, color='steelblue', alpha=0.7)
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title(f'Confidence Score Distribution - {self.model_type.capitalize()} Model')
            plt.axvline(x=np.mean(all_confidences), color='red', linestyle='--', 
                      label=f'Mean Confidence: {np.mean(all_confidences):.4f}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
            plt.close()

########################
# Helper Functions
########################

def test_model_on_images(model_path, image_dir, output_dir='results', conf_threshold=0.25):
    """
    Utility function to test a model on images
    
    Args:
        model_path (str): Path to trained model
        image_dir (str): Directory containing images to test
        output_dir (str): Directory to save results
        conf_threshold (float): Confidence threshold for predictions
        
    Returns:
        list: Test results
    """
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found at {image_dir}")
        return None
    
    # Create tester and run tests
    tester = ModelTester(model_path)
    results = tester.test_on_images(
        image_dir=image_dir,
        output_dir=output_dir,
        conf_threshold=conf_threshold
    )
    
    return results

def create_comparison_chart(primary_results, lightweight_results, output_dir):
    """
    Create comparison charts between primary and lightweight models
    
    Args:
        primary_results (list): Results from primary model
        lightweight_results (list): Results from lightweight model
        output_dir (str): Directory to save comparison charts
    """
    comparison_dir = os.path.join(output_dir, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compare detection counts
    image_names = sorted(list(set([r['image'] for r in primary_results + lightweight_results])))
    
    # Create mapping of image name to detection count
    primary_counts = {r['image']: r['detections'] for r in primary_results}
    lightweight_counts = {r['image']: r['detections'] for r in lightweight_results}
    
    # Get counts for each image
    primary_detections = [primary_counts.get(img, 0) for img in image_names]
    lightweight_detections = [lightweight_counts.get(img, 0) for img in image_names]
    
    # Create grouped bar chart for detections
    plt.figure(figsize=(14, 7))
    x = np.arange(len(image_names))
    width = 0.35
    
    plt.bar(x - width/2, primary_detections, width, label='Primary Model', color='royalblue')
    plt.bar(x + width/2, lightweight_detections, width, label='Lightweight Model', color='lightcoral')
    
    plt.xlabel('Images')
    plt.ylabel('Number of Detections')
    plt.title('Detection Count Comparison: Primary vs Lightweight Model')
    plt.xticks(x, image_names, rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'detection_count_comparison.png'))
    plt.close()
    
    # Compare confidence distributions
    primary_confidences = [conf for result in primary_results for conf in result['confidence']]
    lightweight_confidences = [conf for result in lightweight_results for conf in result['confidence']]
    
    if primary_confidences and lightweight_confidences:
        plt.figure(figsize=(12, 6))
        
        # Plot both histograms
        plt.hist(primary_confidences, bins=20, alpha=0.6, label='Primary Model', color='royalblue')
        plt.hist(lightweight_confidences, bins=20, alpha=0.6, label='Lightweight Model', color='lightcoral')
        
        # Add mean lines
        plt.axvline(x=np.mean(primary_confidences), color='blue', linestyle='--', 
                   label=f'Primary Mean: {np.mean(primary_confidences):.4f}')
        plt.axvline(x=np.mean(lightweight_confidences), color='red', linestyle='--', 
                   label=f'Lightweight Mean: {np.mean(lightweight_confidences):.4f}')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(comparison_dir, 'confidence_comparison.png'))
        plt.close()

def print_model_comparison_stats(primary_results, lightweight_results):
    """
    Print detailed comparison statistics between models
    
    Args:
        primary_results (list): Results from primary model
        lightweight_results (list): Results from lightweight model
    """
    # Size comparison
    if os.path.exists(PRIMARY_MODEL_PATH) and os.path.exists(LIGHTWEIGHT_MODEL_PATH):
        primary_size = os.path.getsize(PRIMARY_MODEL_PATH) / (1024 * 1024)
        light_size = os.path.getsize(LIGHTWEIGHT_MODEL_PATH) / (1024 * 1024)
        size_reduction = (primary_size - light_size) / primary_size * 100
        
        logger.info("\n=== Model Size Comparison ===")
        logger.info(f"Primary Model: {primary_size:.2f} MB")
        logger.info(f"Lightweight Model: {light_size:.2f} MB")
        logger.info(f"Size Reduction: {size_reduction:.2f}%")
    
    # Performance comparison
    if primary_results and lightweight_results:
        # Detections
        primary_total = sum(result['detections'] for result in primary_results)
        lightweight_total = sum(result['detections'] for result in lightweight_results)
        
        # Confidences
        primary_confidences = [conf for result in primary_results for conf in result['confidence']]
        lightweight_confidences = [conf for result in lightweight_results for conf in result['confidence']]
        
        primary_avg_conf = sum(primary_confidences) / len(primary_confidences) if primary_confidences else 0
        lightweight_avg_conf = sum(lightweight_confidences) / len(lightweight_confidences) if lightweight_confidences else 0
        
        # Performance ratios
        detection_ratio = (lightweight_total / primary_total) * 100 if primary_total > 0 else 0
        confidence_ratio = (lightweight_avg_conf / primary_avg_conf) * 100 if primary_avg_conf > 0 else 0
        
        logger.info("\n=== Detection Performance Comparison ===")
        logger.info(f"Primary Model Detections: {primary_total}")
        logger.info(f"Lightweight Model Detections: {lightweight_total}")
        logger.info(f"Lightweight/Primary Detection Ratio: {detection_ratio:.2f}%")
        
        logger.info("\n=== Confidence Performance Comparison ===")
        logger.info(f"Primary Model Average Confidence: {primary_avg_conf:.4f}")
        logger.info(f"Lightweight Model Average Confidence: {lightweight_avg_conf:.4f}")
        logger.info(f"Lightweight/Primary Confidence Ratio: {confidence_ratio:.2f}%")
        
        # Overall performance assessment
        logger.info("\n=== Overall Performance Assessment ===")
        logger.info(f"Size Efficiency: {size_reduction:.2f}% reduction")
        logger.info(f"Detection Efficiency: {detection_ratio:.2f}% of primary model")
        logger.info(f"Confidence Efficiency: {confidence_ratio:.2f}% of primary model")
        
        # Recommendation
        if detection_ratio > 85 and size_reduction > 80:
            logger.info("\nRECOMMENDATION: Lightweight model provides excellent efficiency with minimal performance loss")
        elif detection_ratio > 70 and size_reduction > 70:
            logger.info("\nRECOMMENDATION: Lightweight model provides good balance of efficiency and performance")
        else:
            logger.info("\nRECOMMENDATION: Primary model may be preferred for maximum performance if resources allow")


########################
# Main Execution Functions
########################

def run_unit_tests():
    """Run unit tests for both models"""
    logger.info("=== Running Unit Tests ===")
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(unittest.makeSuite(TestPrimaryModel))
    suite.addTest(unittest.makeSuite(TestLightweightModel))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def run_image_tests(args):
    """Run image tests for models based on arguments"""
    logger.info("=== Running Image Tests ===")
    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Confidence threshold: {args.conf}")
    
    primary_results = None
    lightweight_results = None
    
    # Test primary model
    if args.primary or args.both:
        if os.path.exists(PRIMARY_MODEL_PATH):
            logger.info("\n--- Testing Primary Model ---")
            primary_results = test_model_on_images(
                model_path=PRIMARY_MODEL_PATH,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                conf_threshold=args.conf
            )
        else:
            logger.error(f"Primary model not found at {PRIMARY_MODEL_PATH}")
    
    # Test lightweight model
    if args.lightweight or args.both:
        if os.path.exists(LIGHTWEIGHT_MODEL_PATH):
            logger.info("\n--- Testing Lightweight Model ---")
            lightweight_results = test_model_on_images(
                model_path=LIGHTWEIGHT_MODEL_PATH,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                conf_threshold=args.conf
            )
        else:
            logger.error(f"Lightweight model not found at {LIGHTWEIGHT_MODEL_PATH}")
    
    # Compare models if both were tested
    if primary_results and lightweight_results:
        logger.info("\n--- Comparing Models ---")
        create_comparison_chart(primary_results, lightweight_results, args.output_dir)
        print_model_comparison_stats(primary_results, lightweight_results)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Spill Detection Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unit tests only
  python -m tests.test_spill_detection --unit-tests

  # Run image tests on both models
  python -m tests.test_spill_detection --image-tests --both

  # Test only the primary model
  python -m tests.test_spill_detection --image-tests --primary

  # Test only the lightweight model 
  python -m tests.test_spill_detection --image-tests --lightweight

  # Run all tests (unit tests and image tests on both models)
  python -m tests.test_spill_detection --all
"""
    )
    
    parser.add_argument("--unit-tests", action="store_true",
                      help="Run unit tests on model structure and properties")
    
    parser.add_argument("--image-tests", action="store_true",
                      help="Run tests on images")
    
    parser.add_argument("--all", action="store_true",
                      help="Run all tests (unit tests and image tests on both models)")
    
    # Model selection (for image tests)
    model_group = parser.add_argument_group("Model selection (for image tests)")
    model_args = model_group.add_mutually_exclusive_group()
    model_args.add_argument("--primary", action="store_true",
                          help="Test only the primary model")
    model_args.add_argument("--lightweight", action="store_true",
                          help="Test only the lightweight model")
    model_args.add_argument("--both", action="store_true",
                          help="Test both models (default for image tests)")
    
    # Image test parameters
    image_group = parser.add_argument_group("Image test parameters")
    image_group.add_argument("--image-dir", type=str, default=DEFAULT_TEST_DIR,
                           help=f"Directory containing test images (default: {DEFAULT_TEST_DIR})")
    image_group.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                           help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    image_group.add_argument("--conf", type=float, default=0.25,
                           help="Confidence threshold for detections (default: 0.25)")
    
    args = parser.parse_args()
    
    # Set defaults if not specified
    if args.image_tests and not (args.primary or args.lightweight or args.both):
        args.both = True
    
    if args.all:
        args.unit_tests = True
        args.image_tests = True
        args.both = True
    
    return args


def main():
    """Main function"""
    args = parse_args()
    
    logger.info("=== Spill Detection Comprehensive Testing ===")
    
    # Check if either test type is selected
    if not (args.unit_tests or args.image_tests):
        logger.error("No tests selected. Use --unit-tests, --image-tests, or --all")
        return
    
    # Run unit tests if selected
    if args.unit_tests:
        run_unit_tests()
    
    # Run image tests if selected
    if args.image_tests:
        run_image_tests(args)
    
    logger.info("\n=== Testing Complete ===")


if __name__ == "__main__":
    main() 
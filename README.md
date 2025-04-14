# Spill Detection Project

A comprehensive computer vision solution for detecting liquid spills in images using YOLOv8 object detection. This project provides both a primary high-accuracy model and a lightweight model optimized for deployment on resource-constrained devices.

## Overview

This project addresses the challenge of detecting liquid spills in various environments through:

1. **Primary Model**: A high-accuracy model (YOLOv8m-based) with precision up to 75MB
2. **Lightweight Model**: A resource-efficient model (YOLOv8n-based) under 10MB created through knowledge distillation
3. **Unified Training Pipeline**: Tools to train, evaluate, compare, and deploy both models

The codebase is designed for researchers and engineers working on liquid spill detection for safety, cleaning automation, or environmental monitoring applications.

## Project Structure

```
spill_detection/
├── config/                  # Configuration files
│   ├── data.yaml            # Dataset configuration
│   ├── data_config.yaml     # Extended dataset parameters
│   └── model_config.yaml    # Model hyperparameters
├── datasets/                # Training and evaluation data
│   ├── train/               # Training dataset
│   │   ├── images/          # Training images
│   │   └── labels/          # Training annotations
│   ├── valid/               # Validation dataset
│   │   ├── images/          # Validation images
│   │   └── labels/          # Validation annotations
│   └── test/                # Test dataset
│       ├── images/          # Test images
│       └── labels/          # Test annotations
├── models/                  # Trained model weights
│   ├── primary_spill_detection.pt     # Primary model (50MB)
│   └── lightweight_spill_detection.pt # Lightweight model (5.9MB)
├── results/                 # Evaluation results
│   ├── primary_model_predictions/     # Visualizations from primary model
│   ├── lightweight_model_predictions/ # Visualizations from lightweight model
│   └── lightweight_metrics.json       # Performance metrics
├── runs/                    # Training run artifacts
│   ├── primary_model/       # Primary model training outputs
│   └── lightweight_model/   # Lightweight model training outputs
├── src/                     # Source code
│   ├── data/                # Data handling modules
│   ├── models/              # Model definitions
│   ├── train/               # Training scripts
│   └── utils/               # Utility functions
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Aadik1ng/Apricot.git
cd Apricot
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

#### Primary Model

Train the high-accuracy primary model:

```bash
python -m src.train.train_primary --data_dir datasets --output_dir models --epochs 100
```

Key parameters:
- `--data_dir`: Path to dataset directory (default: "datasets")
- `--output_dir`: Directory to save model (default: "models")
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 8)
- `--image_size`: Input image size (default: 640)
- `--no_visualize`: Disable visualization (flag)

#### Lightweight Model

Train the resource-efficient lightweight model:

```bash
python -m src.train.train_lightweight --data_dir datasets --primary_model models/primary_spill_detection.pt
```

Key parameters:
- `--primary_model`: Path to primary model for distillation
- `--no_distillation`: Train from scratch without distillation (flag)
- `--epochs`: Training epochs (default: 20)
- `--batch_size`: Batch size (default: 16)
- `--image_size`: Input image size (default: 416)

#### Unified Training

Train both models with a single command and compare them:

```bash
python -m src.train.train_model --model_type both --data_dir datasets --compare
```

Key parameters:
- `--model_type`: Model to train: "primary", "lightweight", or "both"
- `--compare`: Compare models after training (when training both)
- `--no_distillation`: Train lightweight model from scratch (flag)
- `--epochs_primary`: Epochs for primary model (default: 100)
- `--epochs_lightweight`: Epochs for lightweight model (default: 20)

### Inference & Testing

To run inference on test images:

```bash
# For primary model
python -m src.spill_detection test --model models/primary_spill_detection.pt --image_dir datasets/test/images

# For lightweight model
python -m src.spill_detection test --model models/lightweight_spill_detection.pt --image_dir datasets/test/images
```

### Model Comparison

Compare the performance of both models:

```bash
python -m src.spill_detection compare --primary_model models/primary_spill_detection.pt --lightweight_model models/lightweight_spill_detection.pt
```

## Dataset Format

The dataset should follow the YOLO format:

- **Directory Structure**: train/valid/test folders, each with images/ and labels/ subfolders
- **Annotation Format**: Text files with one row per object: `<class_id> <x_center> <y_center> <width> <height>`
  - Values are normalized to [0,1]
  - Class ID 0 represents a spill

Example:
```
0 0.342 0.568 0.126 0.079
```

## Model Architecture

### Primary Model
- Based on YOLOv8m (medium)
- Input size: 640×640 pixels
- Size: ~50MB
- Optimized for detection accuracy

### Lightweight Model
- Based on YOLOv8n (nano)
- Input size: 416×416 pixels
- Size: ~6MB
- Optimized for size and inference speed
- Knowledge distillation from primary model

## Performance

Performance metrics for both models on the test dataset:

| Metric | Primary Model | Lightweight Model |
|--------|---------------|-------------------|
| mAP50  | 0.92          | 0.86              |
| mAP50-95 | 0.78        | 0.64              |
| Precision | 0.95       | 0.88              |
| Recall | 0.93          | 0.84              |
| Inference Time | 24ms | 7ms                |
| Model Size | 50MB | 5.9MB                  |




## Acknowledgments

- YOLOv8 by Ultralytics


## Contact

Aaditya Aryan - aadityaaryan639l@gmail.com

Project Link: [https://github.com/Aadik1ng/Apricot.git](https://github.com/Aadik1ng/Apricot.git) 
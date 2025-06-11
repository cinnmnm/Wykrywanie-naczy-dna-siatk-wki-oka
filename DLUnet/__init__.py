"""
DLUnet - Deep Learning U-Net for Retinal Vessel Segmentation

This package provides a complete, modular pipeline for retinal vessel segmentation
using U-Net deep learning architecture. The package is organized into core components
and high-level interfaces for both training and inference.

Core Components:
- model: U-Net architecture implementation
- losses: Specialized loss functions for segmentation
- dataset: Dataset classes for loading and preprocessing
- trainer: Training pipeline and validation logic
- inference: High-performance inference engine

High-Level Interfaces:
- pipeline: Main pipeline class (backward compatible)
- controller: Simple controller interface similar to App\Controller

Usage Examples:

1. Training a new model:
    from DLUnet import VesselSegmentationPipeline
    
    pipeline = VesselSegmentationPipeline("config_unet.yaml")
    pipeline.prepare_data()
    pipeline.initialize_model()
    pipeline.train()

2. Using the controller for inference:
    from DLUnet import VesselSegmentationController
    
    controller = VesselSegmentationController("model.pth")
    results = controller.predict([
        ("image1.jpg", "mask1.png"),
        ("image2.jpg", "mask2.png")
    ])

3. Simple inference function:
    from DLUnet import run_vessel_segmentation
    
    results = run_vessel_segmentation(
        [("image1.jpg", "mask1.png")], 
        "model.pth"
    )
"""

# Import main classes for easy access
from .pipeline import VesselSegmentationPipeline, create_vessel_segmentation_pipeline
from .controller import VesselSegmentationController, create_vessel_segmentation_controller, run_vessel_segmentation

# Import core components
from .Core.model import UNetSegmentation
from .Core.losses import WeightedFocalLoss, DiceLoss, CombinedLoss
from .Core.dataset import VesselSegmentationDataset, SimpleImageDataset
from .Core.trainer import VesselSegmentationTrainer
from .Core.inference import VesselSegmentationInference

# Version info
__version__ = "2.0.0"
__author__ = "Retinal Vessel Segmentation Team"
__description__ = "Deep Learning U-Net for Retinal Vessel Segmentation"

# Define what gets imported with "from DLUnet import *"
__all__ = [
    # Main interfaces
    'VesselSegmentationPipeline',
    'VesselSegmentationController',
    'create_vessel_segmentation_pipeline',
    'create_vessel_segmentation_controller',
    'run_vessel_segmentation',
    
    # Core components
    'UNetSegmentation',
    'WeightedFocalLoss',
    'DiceLoss', 
    'CombinedLoss',
    'VesselSegmentationDataset',
    'SimpleImageDataset',
    'VesselSegmentationTrainer',
    'VesselSegmentationInference'
]

print(f"[DLUnet v{__version__}] Retinal Vessel Segmentation Package Loaded")
print("Available interfaces:")
print("  • VesselSegmentationPipeline - Complete training/inference pipeline")
print("  • VesselSegmentationController - Simple controller interface")
print("  • Core components for custom implementations")

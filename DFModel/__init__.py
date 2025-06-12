"""
DFModel - Decision Forest for Retinal Vessel Classification

This package provides a complete, modular pipeline for retinal vessel classification
using Decision Forest (Random Forest) with patch-based feature extraction.

Core Components:
- model: Decision Forest classifier implementation
- feature_extractor: Advanced patch feature extraction
- dataset: Dataset classes for loading and preprocessing
- trainer: Training pipeline and validation logic
- inference: High-performance inference engine

High-Level Interfaces:
- pipeline: Main pipeline class for training and inference

Usage Examples:

1. Training a new model:
    from DFModel import DecisionForestPipeline
    
    pipeline = DecisionForestPipeline("DFModel/config_dfmodel.yaml")
    pipeline.prepare_data()
    pipeline.initialize_model()
    pipeline.train()

2. Running inference:
    from DFModel import run_decision_forest_classification
    
    results = run_decision_forest_classification(
        [("image1.jpg", "mask1.png")], 
        "model.pkl"
    )

3. Using the pipeline for complete workflow:
    pipeline = DecisionForestPipeline()
    pipeline.prepare_data()
    pipeline.train(hyperparameter_tuning=True)
    pipeline.evaluate()
    predictions = pipeline.predict([("test_image.jpg", "test_mask.png")])
"""

# Import main classes for easy access
from .pipeline import DecisionForestPipeline, create_decision_forest_pipeline, run_decision_forest_classification
from .controller import DecisionForestController, create_decision_forest_controller

# Import core components
from .Core.model import DecisionForestClassifier
from .Core.feature_extractor import PatchFeatureExtractor
from .Core.dataset import PatchDataset
from .Core.trainer import DecisionForestTrainer
from .Core.inference import DecisionForestInference

# Version info
__version__ = "1.0.0"
__author__ = "Decision Forest Team"
__description__ = "Decision Forest for Retinal Vessel Classification using Patch Features"

# Define what gets imported with "from DFModel import *"
__all__ = [
    # Main interfaces
    'DecisionForestPipeline',
    'DecisionForestController',
    'create_decision_forest_pipeline',
    'create_decision_forest_controller',
    'run_decision_forest_classification',
    
    # Core components
    'DecisionForestClassifier',
    'PatchFeatureExtractor',
    'PatchDataset',
    'DecisionForestTrainer',
    'DecisionForestInference'
]

print(f"[DFModel v{__version__}] Decision Forest Vessel Classification Package Loaded")
print("Available interfaces:")
print("  • DecisionForestPipeline - Complete training/inference pipeline")
print("  • DecisionForestController - Simple controller interface")
print("  • Core components for custom implementations")
print("  • Feature extraction with configurable patch strategies")

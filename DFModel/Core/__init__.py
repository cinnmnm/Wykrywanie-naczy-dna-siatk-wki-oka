"""
DFModel Core - Decision Forest for Retinal Vessel Classification

This package provides core components for decision forest-based vessel classification
using patch-based feature extraction.

Core Components:
- model: Decision Forest classifier implementation
- dataset: Patch dataset classes with feature extraction
- trainer: Training pipeline for decision forest
- feature_extractor: Advanced feature extraction from patches
- inference: High-performance inference engine for patch classification
"""

from .model import DecisionForestClassifier
from .dataset import PatchDataset, ImageDataset
from .trainer import DecisionForestTrainer
from .feature_extractor import PatchFeatureExtractor
from .inference import DecisionForestInference

__version__ = "1.0.0"
__author__ = "Decision Forest Team"
__description__ = "Decision Forest for Retinal Vessel Classification using Patch Features"

__all__ = [
    'DecisionForestClassifier',
    'PatchDataset',
    'ImageDataset', 
    'DecisionForestTrainer',
    'PatchFeatureExtractor',
    'DecisionForestInference'
]

print(f"[DFModel v{__version__}] Decision Forest Vessel Classification Package Loaded")

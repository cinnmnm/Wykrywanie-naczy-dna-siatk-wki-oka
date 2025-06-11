"""
Comprehensive Vessel Segmentation Pipeline

This module provides the main pipeline class that combines training and inference
capabilities with a clean, modular architecture. It maintains backward compatibility
while providing improved organization.
"""

import os
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from PIL import Image

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .Core.trainer_clean import VesselSegmentationTrainer
from .Core.inference import VesselSegmentationInference
from .controller import VesselSegmentationController
from Util.config import Config


class VesselSegmentationPipeline:
    """
    Main pipeline class for retinal vessel segmentation.
    
    This class provides a unified interface for both training and inference,
    maintaining the same API as the original VesselSegmentationPipeline but
    with improved modular architecture.
    """
    
    def __init__(self, config_path: str = "config_unet.yaml"):
        """
        Initialize the segmentation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = Config.load(config_path)
        
        # Initialize components
        self.trainer = VesselSegmentationTrainer(config_path)
        self.inference_engine = None
        self.controller = None
        
        print(f"[VesselSegmentationPipeline] Initialized with config: {config_path}")
    
    # Training Interface (delegates to trainer)
    def prepare_data(self, train_split: float = 0.7, val_split: float = 0.15, test_split: float = 0.15):
        """Prepare datasets and data loaders for training"""
        return self.trainer.prepare_data(train_split, val_split, test_split)
    
    def initialize_model(self, in_channels: int = 3, num_classes: int = 2, base_features: int = 64):
        """Initialize the U-Net model"""
        return self.trainer.initialize_model(in_channels, num_classes, base_features)
    
    def train(self, num_epochs: int = None, save_path: str = None):
        """Train the model"""
        return self.trainer.train(num_epochs, save_path)
    
    def evaluate(self, loader=None):
        """Evaluate the model on test set"""
        return self.trainer.evaluate(loader)
    
    def load_weights(self, path: str):
        """Load model weights from file"""
        return self.trainer.load_weights(path)
    
    def plot_training_history(self):
        """Plot training history"""
        return self.trainer.plot_training_history()
    
    # Inference Interface
    def predict(self, 
               image_paths: Union[List[str], List[Tuple]], 
               output_dir: str = None, 
               threshold: float = 0.5,
               **kwargs):
        """
        Run inference on new images.
        
        This method supports both the old API (list of image paths) and new API (list of tuples).
        
        Args:
            image_paths: List of image paths OR list of (image, mask) tuples
            output_dir: Directory to save prediction results (optional)
            threshold: Threshold for binary predictions
            **kwargs: Additional arguments
        
        Returns:
            List of prediction results
        """
        # Determine if using old API (image paths) or new API (image-mask tuples)
        if not image_paths:
            return []
        
        first_item = image_paths[0]
        
        if isinstance(first_item, (tuple, list)) and len(first_item) == 2:
            # New API: list of (image, mask) tuples
            return self._predict_with_tuples(image_paths, output_dir, threshold, **kwargs)
        else:
            # Old API: list of image paths
            return self._predict_with_paths(image_paths, output_dir, threshold, **kwargs)
    
    def _predict_with_paths(self, 
                           image_paths: List[str], 
                           output_dir: str = None, 
                           threshold: float = 0.5,
                           **kwargs):
        """Handle old API with image paths"""
        if self.inference_engine is None:
            model_path = self.config.get('model_save_path', 'DLUnet/SavedModels/best_model.pth')
            if not os.path.exists(model_path):
                raise ValueError(f"Model not found at {model_path}. Train model first or specify correct path.")
            
            self.inference_engine = VesselSegmentationInference(
                model_path=model_path,
                config_path=self.config_path
            )
        
        # Convert to (image, mask) pairs by trying to find corresponding masks
        image_mask_pairs = []
        for img_path in image_paths:
            mask_path = self._find_mask_path(img_path)
            image_mask_pairs.append((img_path, mask_path))
        
        # Use the new inference engine
        predictions = self.inference_engine.predict_batch(
            image_mask_pairs=image_mask_pairs,
            threshold=threshold,
            return_probability=False
        )
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_paths = []
            
            for img_path, prediction in zip(image_paths, predictions):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Save binary mask
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                binary_img = (prediction * 255).astype(np.uint8)
                Image.fromarray(binary_img).save(mask_path)
                output_paths.append(mask_path)
            
            print(f"[VesselSegmentationPipeline] Results saved to: {output_dir}")
        
        return predictions
    
    def _predict_with_tuples(self, 
                            image_mask_pairs: List[Tuple], 
                            output_dir: str = None, 
                            threshold: float = 0.5,
                            **kwargs):
        """Handle new API with image-mask tuples"""
        if self.controller is None:
            model_path = self.config.get('model_save_path', 'DLUnet/SavedModels/best_model.pth')
            if not os.path.exists(model_path):
                raise ValueError(f"Model not found at {model_path}. Train model first or specify correct path.")
            
            self.controller = VesselSegmentationController(
                model_path=model_path,
                config_path=self.config_path
            )
        
        # Use the controller for prediction
        predictions = self.controller.predict(
            image_mask_pairs=image_mask_pairs,
            threshold=threshold,
            return_probability=False
        )
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            for i, (pair, prediction) in enumerate(zip(image_mask_pairs, predictions)):
                image, mask = pair
                
                # Generate base name
                if isinstance(image, str):
                    base_name = os.path.splitext(os.path.basename(image))[0]
                else:
                    base_name = f"image_{i:03d}"
                
                # Save binary mask
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                binary_img = (prediction * 255).astype(np.uint8)
                Image.fromarray(binary_img).save(mask_path)
            
            print(f"[VesselSegmentationPipeline] Results saved to: {output_dir}")
        
        return predictions
    
    def _find_mask_path(self, image_path: str) -> Optional[str]:
        """
        Find corresponding mask file for an image path.
        Assumes mask files are in a 'mask' directory with '_mask' suffix.
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path)
        
        # Try different possible mask locations
        possible_paths = [
            # Same directory with _mask suffix
            os.path.join(image_dir, f"{base_name}_mask.tif"),
            os.path.join(image_dir, f"{base_name}_mask.png"),
            # Mask subdirectory
            os.path.join(os.path.dirname(image_dir), "mask", f"{base_name}_mask.tif"),
            os.path.join(os.path.dirname(image_dir), "mask", f"{base_name}_mask.png"),
            # Adjacent mask directory
            os.path.join(image_dir, "..", "mask", f"{base_name}_mask.tif"),
            os.path.join(image_dir, "..", "mask", f"{base_name}_mask.png"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    # Convenience properties
    @property
    def device(self):
        """Get the device being used"""
        return self.trainer.device
    
    @property
    def model(self):
        """Get the underlying model"""
        return self.trainer.model
    
    @property
    def train_loader(self):
        """Get the training data loader"""
        return self.trainer.train_loader
    
    @property
    def val_loader(self):
        """Get the validation data loader"""
        return self.trainer.val_loader
    
    @property
    def test_loader(self):
        """Get the test data loader"""
        return self.trainer.test_loader
    
    @property
    def training_history(self):
        """Get the training history"""
        return self.trainer.training_history


# Factory function for easy instantiation
def create_vessel_segmentation_pipeline(config_path: str = "config_unet.yaml") -> VesselSegmentationPipeline:
    """
    Factory function to create a vessel segmentation pipeline.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Initialized VesselSegmentationPipeline
    """
    return VesselSegmentationPipeline(config_path)

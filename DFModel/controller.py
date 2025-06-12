"""
Decision Forest Controller

This module provides a high-level interface for decision forest-based
retinal vessel classification, similar to the App\Controller pattern.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .Core.inference import DecisionForestInference
from .pipeline import DecisionForestPipeline


class DecisionForestController:
    """
    High-level controller for decision forest vessel classification.
    
    This class provides a simple interface for running inference
    similar to the existing App\Controller pattern.
    """
    
    def __init__(self, model_path: str, config_path: str = "DFModel/config_dfmodel.yaml"):
        """
        Initialize the controller.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Initialize inference engine
        self.inference_engine = DecisionForestInference(model_path, config_path)
        
        print(f"[DecisionForestController] Initialized with model: {model_path}")
    
    def predict(self, image_mask_pairs: List[Tuple]) -> List[np.ndarray]:
        """
        Run vessel classification on image-mask pairs.
        
        Args:
            image_mask_pairs: List of (image_path, mask_path) tuples
            
        Returns:
            List of prediction maps
        """
        print(f"[DecisionForestController] Running prediction on {len(image_mask_pairs)} images")
        
        # Extract paths
        image_paths = [pair[0] for pair in image_mask_pairs]
        mask_paths = [pair[1] if len(pair) > 1 else None for pair in image_mask_pairs]
        
        # Run inference
        predictions = self.inference_engine.predict_from_paths(image_paths, mask_paths)
        
        return predictions
    
    def predict_single(self, image_path: str, mask_path: Optional[str] = None) -> np.ndarray:
        """
        Run vessel classification on a single image.
        
        Args:
            image_path: Path to input image
            mask_path: Optional path to mask
            
        Returns:
            Prediction map
        """
        return self.predict([(image_path, mask_path)])[0]
    
    def predict_with_postprocessing(self, image_mask_pairs: List[Tuple],
                                   threshold: float = 0.5,
                                   min_area: int = 50) -> List[np.ndarray]:
        """
        Run vessel classification with postprocessing.
        
        Args:
            image_mask_pairs: List of (image_path, mask_path) tuples
            threshold: Classification threshold
            min_area: Minimum area for connected components
            
        Returns:
            List of binary prediction masks
        """
        # Get raw predictions
        predictions = self.predict(image_mask_pairs)
        
        # Apply postprocessing
        processed_predictions = []
        for pred in predictions:
            processed_pred = self.inference_engine.postprocess_predictions(
                pred, min_area=min_area
            )
            processed_predictions.append(processed_pred)
        
        return processed_predictions
    
    def save_predictions(self, image_mask_pairs: List[Tuple], 
                        output_dir: str,
                        save_binary: bool = True,
                        save_probability: bool = False) -> None:
        """
        Run inference and save results to files.
        
        Args:
            image_mask_pairs: List of (image_path, mask_path) tuples
            output_dir: Output directory
            save_binary: Whether to save binary masks
            save_probability: Whether to save probability maps
        """
        # Run inference
        predictions = self.predict(image_mask_pairs)
        
        # Extract image names
        image_names = [os.path.basename(pair[0]) for pair in image_mask_pairs]
        
        # Save results
        self.inference_engine.save_predictions(
            predictions, output_dir, image_names,
            save_binary=save_binary, save_probability=save_probability
        )
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return self.inference_engine.get_model_info()


# Convenience functions for backward compatibility
def create_decision_forest_controller(model_path: str, 
                                    config_path: str = "DFModel/config_dfmodel.yaml") -> DecisionForestController:
    """
    Create a decision forest controller instance.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        
    Returns:
        DecisionForestController instance
    """
    return DecisionForestController(model_path, config_path)


def run_decision_forest_classification(image_mask_pairs: List[Tuple], 
                                     model_path: str,
                                     config_path: str = "DFModel/config_dfmodel.yaml") -> List[np.ndarray]:
    """
    Run decision forest classification on image-mask pairs.
    
    Args:
        image_mask_pairs: List of (image_path, mask_path) tuples
        model_path: Path to trained model
        config_path: Path to configuration file
        
    Returns:
        List of prediction maps
    """
    controller = DecisionForestController(model_path, config_path)
    return controller.predict(image_mask_pairs)

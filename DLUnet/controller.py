"""
Vessel Segmentation Controller

This module provides a high-level interface for retinal vessel segmentation
similar to the App\Controller pattern. It wraps the core segmentation
functionality into an easy-to-use API.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .Core.inference import VesselSegmentationInference


class VesselSegmentationController:
    """
    High-level controller for vessel segmentation.
    
    Provides a simple interface similar to App\Controller for performing
    vessel segmentation on retinal images with optional masks.
    
    Usage:
        controller = VesselSegmentationController("path/to/model.pth")
        results = controller.predict([(image1, mask1), (image2, mask2)])
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "DLUnet/config_unet.yaml",
                 device: str = None):
        """
        Initialize the vessel segmentation controller.
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.inference_engine = VesselSegmentationInference(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        
        print("[VesselSegmentationController] Controller initialized and ready!")
    
    def predict(self, 
               image_mask_pairs: List[Tuple[Union[str, np.ndarray, Image.Image], 
                                          Optional[Union[str, np.ndarray, Image.Image]]]],
               threshold: float = 0.5,
               return_probability: bool = False) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Perform vessel segmentation on a list of image-mask pairs.
        
        This is the main API method that takes a list of (image, mask) tuples
        and returns segmentation results. Similar to FilterSegmentation.run()
        but for deep learning-based vessel segmentation.
        
        Args:
            image_mask_pairs: List of (image, mask) tuples where:
                - image: Can be file path (str), numpy array, or PIL Image
                - mask: Optional mask (same types as image), can be None
            threshold: Threshold for binary segmentation (0.0-1.0)
            return_probability: If True, returns (binary_mask, probability_map) tuples
        
        Returns:
            List of segmentation results:
                - If return_probability=False: List of binary masks (numpy arrays)
                - If return_probability=True: List of (binary_mask, probability_map) tuples
        
        Example:
            controller = VesselSegmentationController("model.pth")
            
            # Process images with masks
            pairs = [
                ("image1.jpg", "mask1.png"),
                ("image2.jpg", "mask2.png"),
                (image_array, mask_array),
                (pil_image, None)  # No mask
            ]
            
            results = controller.predict(pairs)
            # results[i] is the segmentation for pairs[i]
        """
        print(f"[VesselSegmentationController] Processing {len(image_mask_pairs)} image-mask pairs...")
        
        # Validate inputs
        validated_pairs = []
        for i, pair in enumerate(image_mask_pairs):
            if len(pair) != 2:
                raise ValueError(f"Item {i} must be a (image, mask) tuple, got {len(pair)} elements")
            
            image, mask = pair
            
            # Validate image
            if not isinstance(image, (str, np.ndarray, Image.Image)):
                raise ValueError(f"Image {i} must be str, numpy array, or PIL Image, got {type(image)}")
            
            # Validate mask (can be None)
            if mask is not None and not isinstance(mask, (str, np.ndarray, Image.Image)):
                raise ValueError(f"Mask {i} must be str, numpy array, PIL Image, or None, got {type(mask)}")
            
            validated_pairs.append((image, mask))
        
        # Perform batch inference
        results = self.inference_engine.predict_batch(
            image_mask_pairs=validated_pairs,
            threshold=threshold,
            return_probability=return_probability,
            batch_size=4  # Configurable batch size
        )
        
        print(f"[VesselSegmentationController] Segmentation complete!")
        return results
    
    def predict_single(self,
                      image: Union[str, np.ndarray, Image.Image],
                      mask: Optional[Union[str, np.ndarray, Image.Image]] = None,
                      threshold: float = 0.5,
                      return_probability: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform vessel segmentation on a single image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            mask: Optional mask (same types as image)
            threshold: Threshold for binary segmentation
            return_probability: If True, returns (binary_mask, probability_map)
        
        Returns:
            Binary segmentation mask or (binary_mask, probability_map) if return_probability=True
        
        Example:
            controller = VesselSegmentationController("model.pth")
            
            # Process single image
            binary_mask = controller.predict_single("retinal_image.jpg", "mask.png")
            
            # With probability map
            binary_mask, prob_map = controller.predict_single(
                "retinal_image.jpg", 
                "mask.png", 
                return_probability=True
            )
        """
        return self.inference_engine.predict_single(
            image=image,
            mask=mask,
            threshold=threshold,
            return_probability=return_probability
        )
    
    def run_dl_segmentation(self, 
                           image: Union[str, np.ndarray, Image.Image],
                           mask: Optional[Union[str, np.ndarray, Image.Image]] = None) -> np.ndarray:
        """
        Simple interface matching the App\Controller.run_dl() pattern.
        
        Args:
            image: Input retinal image
            mask: Optional mask for valid regions
        
        Returns:
            Binary vessel segmentation mask (numpy array)
        
        Example:
            controller = VesselSegmentationController("model.pth")
            segmentation = controller.run_dl_segmentation(image, mask)
        """
        return self.predict_single(image, mask, threshold=0.5, return_probability=False)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model and configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'device': str(self.inference_engine.device),
            'target_size': self.inference_engine.target_size,
            'global_contrast_normalization': self.inference_engine.global_contrast_norm,
            'model_type': 'U-Net',
            'input_channels': 3,
            'output_classes': 2
        }
    
    def set_threshold(self, threshold: float):
        """
        Set default threshold for binary predictions.
        
        Args:
            threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.default_threshold = threshold
        print(f"[VesselSegmentationController] Default threshold set to {threshold}")


# Convenience functions for backward compatibility
def create_vessel_segmentation_controller(model_path: str, 
                                        config_path: str = "DLUnet/config_unet.yaml") -> VesselSegmentationController:
    """
    Factory function to create a vessel segmentation controller.
    
    Args:
        model_path: Path to trained model weights
        config_path: Path to configuration file
    
    Returns:
        Initialized VesselSegmentationController
    """
    return VesselSegmentationController(model_path, config_path)


def run_vessel_segmentation(image_mask_pairs: List[Tuple], 
                          model_path: str,
                          config_path: str = "DLUnet/config_unet.yaml") -> List[np.ndarray]:
    """
    One-shot function for vessel segmentation.
    
    Args:
        image_mask_pairs: List of (image, mask) tuples
        model_path: Path to trained model weights
        config_path: Path to configuration file
    
    Returns:
        List of binary segmentation masks
    """
    controller = VesselSegmentationController(model_path, config_path)
    return controller.predict(image_mask_pairs)

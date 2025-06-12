"""
Decision Forest Inference Engine

This module provides high-performance inference capabilities
for decision forest-based vessel classification.
"""

import numpy as np
import cv2
from PIL import Image
import os
from typing import List, Tuple, Optional, Dict, Union
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Util.config import Config
from .model import DecisionForestClassifier
from .feature_extractor import PatchFeatureExtractor


class DecisionForestInference:
    """
    High-performance inference engine for Decision Forest vessel classification.
    
    This class provides methods for running inference on single images
    or batches of images using trained decision forest models.
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model file
            config_path: Optional path to configuration file
        """
        self.model_path = model_path
        self.config = Config(config_path).get_config() if config_path else {}
        
        # Load model
        self.model = DecisionForestClassifier(self.config)
        self.model.load_model(model_path)
        
        # Initialize feature extractor
        self.feature_extractor = PatchFeatureExtractor(self.config)
        
        # Inference configuration
        self.patch_size = self.config.get('patch', {}).get('size', 27)
        self.stride = self.config.get('inference', {}).get('stride', 1)  # Sliding window stride
        self.batch_size = self.config.get('inference', {}).get('batch_size', 1000)  # Feature batch size
        
        print(f"[DecisionForestInference] Initialized")
        print(f"  Model: {model_path}")
        print(f"  Patch size: {self.patch_size}x{self.patch_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Batch size: {self.batch_size}")
    
    def predict_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict vessel segmentation for a single image.
        
        Args:
            image: Input image of shape (H, W, C)
            mask: Optional mask to limit inference region
            
        Returns:
            Prediction map of shape (H, W) with vessel probabilities
        """
        print(f"[DecisionForestInference] Predicting image of shape: {image.shape}")
        start_time = time.time()
        
        # Extract patches and positions
        patches, positions = self._extract_sliding_window_patches(image, mask)
        
        if len(patches) == 0:
            print("Warning: No valid patches found")
            return np.zeros(image.shape[:2])
        
        print(f"[DecisionForestInference] Extracted {len(patches)} patches")
        
        # Extract features in batches
        predictions = []
        num_batches = (len(patches) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(patches))
            batch_patches = patches[start_idx:end_idx]
            
            # Extract features for batch
            batch_features = []
            for patch in batch_patches:
                features = self.feature_extractor.extract_comprehensive_features(patch)
                batch_features.append(features)
            
            batch_features = np.array(batch_features)
            
            # Predict probabilities
            batch_probs = self.model.predict_proba(batch_features)
            predictions.extend(batch_probs[:, 1])  # Vessel probability
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")
        
        # Create prediction map
        prediction_map = self._create_prediction_map(
            image.shape[:2], positions, predictions
        )
        
        inference_time = time.time() - start_time
        print(f"[DecisionForestInference] Inference completed in {inference_time:.2f} seconds")
        
        return prediction_map
    
    def predict_batch(self, images: List[np.ndarray], 
                     masks: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Predict vessel segmentation for a batch of images.
        
        Args:
            images: List of input images
            masks: Optional list of masks
            
        Returns:
            List of prediction maps
        """
        print(f"[DecisionForestInference] Predicting batch of {len(images)} images")
        
        if masks is None:
            masks = [None] * len(images)
        
        predictions = []
        for i, (image, mask) in enumerate(zip(images, masks)):
            print(f"Processing image {i + 1}/{len(images)}")
            pred = self.predict_image(image, mask)
            predictions.append(pred)
        
        return predictions
    
    def predict_from_paths(self, image_paths: List[str], 
                          mask_paths: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        Predict vessel segmentation from image file paths.
        
        Args:
            image_paths: List of image file paths
            mask_paths: Optional list of mask file paths
            
        Returns:
            List of prediction maps
        """
        print(f"[DecisionForestInference] Loading and predicting {len(image_paths)} images")
        
        # Load images
        images = []
        masks = []
        
        for i, img_path in enumerate(image_paths):
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            images.append(image)
            
            # Load mask if provided
            if mask_paths and i < len(mask_paths):
                mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 127).astype(np.uint8)
                masks.append(mask)
            else:
                masks.append(None)
        
        # Predict
        return self.predict_batch(images, masks if any(m is not None for m in masks) else None)
    
    def _extract_sliding_window_patches(self, image: np.ndarray, 
                                       mask: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches using sliding window approach.
        
        Args:
            image: Input image
            mask: Optional mask
            
        Returns:
            Tuple of (patches, positions)
        """
        patches = []
        positions = []
        
        half_patch = self.patch_size // 2
        h, w = image.shape[:2]
        
        # Create default mask if none provided
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8)
        
        # Extract patches with sliding window
        for y in range(half_patch, h - half_patch, self.stride):
            for x in range(half_patch, w - half_patch, self.stride):
                # Check if patch center is in valid region
                if mask[y, x] > 0:
                    # Extract patch
                    patch = image[y - half_patch:y + half_patch + 1, 
                                x - half_patch:x + half_patch + 1]
                    
                    # Check if patch has valid size
                    if patch.shape[:2] == (self.patch_size, self.patch_size):
                        patches.append(patch)
                        positions.append((y, x))
        
        return patches, positions
    
    def _create_prediction_map(self, image_shape: Tuple[int, int], 
                              positions: List[Tuple[int, int]], 
                              predictions: List[float]) -> np.ndarray:
        """
        Create prediction map from patch predictions.
        
        Args:
            image_shape: Shape of original image (H, W)
            positions: List of patch center positions
            predictions: List of prediction values
            
        Returns:
            Prediction map
        """
        h, w = image_shape
        prediction_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)
        
        half_patch = self.patch_size // 2
        
        # Accumulate predictions
        for (y, x), pred in zip(positions, predictions):
            # Add prediction to patch region
            y_start = max(0, y - half_patch)
            y_end = min(h, y + half_patch + 1)
            x_start = max(0, x - half_patch)
            x_end = min(w, x + half_patch + 1)
            
            prediction_map[y_start:y_end, x_start:x_end] += pred
            count_map[y_start:y_end, x_start:x_end] += 1
        
        # Average overlapping predictions
        valid_mask = count_map > 0
        prediction_map[valid_mask] /= count_map[valid_mask]
        
        return prediction_map
    
    def create_binary_mask(self, prediction_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Create binary mask from prediction map.
        
        Args:
            prediction_map: Prediction probability map
            threshold: Classification threshold
            
        Returns:
            Binary mask (0 = background, 255 = vessel)
        """
        binary_mask = (prediction_map > threshold).astype(np.uint8) * 255
        return binary_mask
    
    def postprocess_predictions(self, prediction_map: np.ndarray, 
                               min_area: int = 50,
                               morphology_kernel_size: int = 3) -> np.ndarray:
        """
        Apply postprocessing to prediction map.
        
        Args:
            prediction_map: Raw prediction map
            min_area: Minimum area for connected components
            morphology_kernel_size: Size of morphological operations kernel
            
        Returns:
            Postprocessed prediction map
        """
        # Create binary mask
        binary_mask = self.create_binary_mask(prediction_map)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (morphology_kernel_size, morphology_kernel_size))
        
        # Opening to remove noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill gaps
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small components
        if min_area > 0:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
            
            # Filter by area
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area < min_area:
                    binary_mask[labels == label] = 0
        
        return binary_mask
    
    def save_predictions(self, prediction_maps: List[np.ndarray], 
                        output_dir: str, 
                        image_names: Optional[List[str]] = None,
                        save_binary: bool = True,
                        save_probability: bool = True) -> None:
        """
        Save prediction results to files.
        
        Args:
            prediction_maps: List of prediction maps
            output_dir: Output directory
            image_names: Optional list of image names
            save_binary: Whether to save binary masks
            save_probability: Whether to save probability maps
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if image_names is None:
            image_names = [f"prediction_{i:04d}" for i in range(len(prediction_maps))]
        
        print(f"[DecisionForestInference] Saving {len(prediction_maps)} predictions to: {output_dir}")
        
        for i, (pred_map, name) in enumerate(zip(prediction_maps, image_names)):
            base_name = os.path.splitext(name)[0]
            
            if save_probability:
                # Save probability map as floating point image
                prob_path = os.path.join(output_dir, f"{base_name}_prob.tiff")
                cv2.imwrite(prob_path, (pred_map * 255).astype(np.uint8))
            
            if save_binary:
                # Save binary mask
                binary_mask = self.create_binary_mask(pred_map)
                binary_path = os.path.join(output_dir, f"{base_name}_binary.png")
                cv2.imwrite(binary_path, binary_mask)
        
        print(f"[DecisionForestInference] Predictions saved successfully")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        model_info = self.model.get_model_info()
        
        inference_info = {
            'patch_size': self.patch_size,
            'stride': self.stride,
            'batch_size': self.batch_size,
            'model_path': self.model_path
        }
        
        return {**model_info, **inference_info}

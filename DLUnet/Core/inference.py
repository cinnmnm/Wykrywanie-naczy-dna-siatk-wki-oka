"""
Inference Engine for Vessel Segmentation

This module provides fast and efficient inference capabilities
for retinal vessel segmentation using trained U-Net models.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
import sys

from .model import UNetSegmentation
from .dataset import SimpleImageDataset

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Data.Preprocessing import ImagePreprocessing
from Util.config import Config


class VesselSegmentationInference:
    """
    High-performance inference engine for vessel segmentation.
    
    Provides both single image and batch processing capabilities
    with mask-aware predictions and flexible output options.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config.yaml",
                 device: Optional[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to configuration file
            device: Device to use ('cuda', 'cpu', or None for auto)
        """        # Initialize with proper type hints
        self.model: Optional[UNetSegmentation] = None
        self.config: Dict = {}
        self.device: torch.device
        self.target_size: Tuple[int, int]
        self.global_contrast_norm: bool
        self.transform = None  # Will be torchvision.transforms.Compose
        
        # Load configuration
        self._load_config(config_path)
        
        # Setup device
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Model configuration
        self.target_size = tuple(self.config.get('resize_shape', [512, 512]))
        self.global_contrast_norm = self.config.get('preprocessing', {}).get('global_contrast_normalization', False)
        
        # Initialize and load model
        self._initialize_model(model_path)
        
        # Create transforms
        self.transform = self._create_transforms()
        
        print(f"[VesselSegmentationInference] Initialized")
        print(f"  Device: {self.device}")
        print(f"  Model loaded from: {model_path}")
        print(f"  Target size: {self.target_size}")
        print(f"  Global contrast normalization: {self.global_contrast_norm}")
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration with proper error handling"""
        try:
            self.config = Config.load(config_path)
            if self.config is None:
                raise ValueError(f"Failed to load config from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration...")
            self.config = {
                'resize_shape': [512, 512],
                'model': {'base_features': 32},  # Default to demo model size
                'preprocessing': {'global_contrast_normalization': False}
            }
    
    def _initialize_model(self, model_path: str) -> None:
        """Initialize and load the model"""
        # Initialize model
        model_config = self.config.get('model', {})
        base_features = model_config.get('base_features', 32)  # Default to demo model size
        
        self.model = UNetSegmentation(
            in_channels=3,
            num_classes=2,
            base_features=base_features
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
    
    def _ensure_model_ready(self) -> None:
        """Ensure model and transforms are initialized"""
        if self.model is None:
            raise RuntimeError("Model must be initialized before inference")
        if self.transform is None:
            raise RuntimeError("Transform must be initialized before inference")
    
    def _create_transforms(self):
        """Create image transforms for inference"""
        from torchvision import transforms as T
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict_single(self, 
                      image: Union[str, Image.Image, np.ndarray],
                      mask: Optional[Union[str, Image.Image, np.ndarray]] = None,
                      threshold: float = 0.5,
                      return_probability: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict vessel segmentation for a single image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            mask: Optional mask (path, PIL Image, or numpy array)
            threshold: Threshold for binary predictions
            return_probability: Whether to return probability map along with binary
        
        Returns:
            Binary segmentation mask or (binary_mask, probability_map) if return_probability=True
        """
        # Ensure model and transforms are ready
        self._ensure_model_ready()
        
        # Type assertions to help the type checker
        assert self.model is not None, "Model must be initialized"
        assert self.transform is not None, "Transform must be initialized"
        
        # Load and preprocess image
        if isinstance(image, str):
            image_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image.convert('RGB') if image.mode != 'RGB' else image
        
        original_size = image_pil.size  # (W, H)
        
        # Resize image
        image_resized = image_pil.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Apply global contrast normalization if enabled
        if self.global_contrast_norm:
            image_np = np.array(image_resized)
            image_gcn = ImagePreprocessing.global_contrast_normalization(image_np)
            image_gcn = np.clip(
                (image_gcn - image_gcn.min()) / (image_gcn.max() - image_gcn.min() + 1e-8) * 255, 
                0, 255
            ).astype(np.uint8)
            image_resized = Image.fromarray(image_gcn)
        
        # Transform image
        image_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
        
        # Process mask if provided
        mask_tensor = None
        if mask is not None:
            if isinstance(mask, str):
                mask_pil = Image.open(mask).convert('L')
            elif isinstance(mask, np.ndarray):
                mask_pil = Image.fromarray(mask)
            else:
                mask_pil = mask.convert('L') if mask.mode != 'L' else mask
            
            mask_resized = mask_pil.resize(self.target_size, Image.Resampling.NEAREST)
            mask_np = (np.array(mask_resized) > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.softmax(output, dim=1)[0, 1]  # Vessel probability
            
            # Apply mask if available
            if mask_tensor is not None:
                probability = probability * mask_tensor
            
            # Convert back to original size
            probability_resized = TF.resize(
                probability.unsqueeze(0), 
                [original_size[1], original_size[0]]  # PIL uses (W,H), torch uses (H,W)
            ).squeeze(0)
            
            probability_np = probability_resized.cpu().numpy()
            binary_mask = (probability_np > threshold).astype(np.uint8)
        
        if return_probability:
            return binary_mask, probability_np
        else:
            return binary_mask
    
    def predict_batch(self,
                     image_mask_pairs: List[Tuple[Union[str, Image.Image, np.ndarray], Optional[Union[str, Image.Image, np.ndarray]]]],
                     threshold: float = 0.5,
                     batch_size: int = 4,
                     return_probability: bool = False) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Predict vessel segmentation for multiple image-mask pairs.
        
        Args:
            image_mask_pairs: List of (image, mask) tuples
            threshold: Threshold for binary predictions
            batch_size: Batch size for processing
            return_probability: Whether to return probability maps
        
        Returns:
            List of predictions (binary masks or tuples if return_probability=True)
        """
        # Ensure model and transforms are ready
        self._ensure_model_ready()
        
        print(f"[VesselSegmentationInference] Processing {len(image_mask_pairs)} images...")
        
        results = []
        
        for i in range(0, len(image_mask_pairs), batch_size):
            batch_pairs = image_mask_pairs[i:i+batch_size]
            batch_results = self._process_batch(batch_pairs, threshold, return_probability)
            results.extend(batch_results)
        
        print(f"[VesselSegmentationInference] Batch processing complete!")
        return results
    
    def _process_batch(self,
                      batch_pairs: List[Tuple],
                      threshold: float,
                      return_probability: bool) -> List:
        """Process a single batch of image-mask pairs"""
        # Type assertions to help the type checker
        assert self.model is not None, "Model must be initialized"
        assert self.transform is not None, "Transform must be initialized"
        
        # Prepare batch data
        images = []
        masks = []
        original_sizes = []
        
        for image, mask in batch_pairs:
            # Load and preprocess image
            if isinstance(image, str):
                image_pil = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image.convert('RGB') if image.mode != 'RGB' else image
            
            original_sizes.append(image_pil.size)  # (W, H)
            
            # Resize and transform
            image_resized = image_pil.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Apply GCN if enabled
            if self.global_contrast_norm:
                image_np = np.array(image_resized)
                image_gcn = ImagePreprocessing.global_contrast_normalization(image_np)
                image_gcn = np.clip(
                    (image_gcn - image_gcn.min()) / (image_gcn.max() - image_gcn.min() + 1e-8) * 255, 
                    0, 255
                ).astype(np.uint8)
                image_resized = Image.fromarray(image_gcn)
            
            image_tensor = self.transform(image_resized)
            images.append(image_tensor)
            
            # Process mask
            if mask is not None:
                if isinstance(mask, str):
                    mask_pil = Image.open(mask).convert('L')
                elif isinstance(mask, np.ndarray):
                    mask_pil = Image.fromarray(mask)
                else:
                    mask_pil = mask.convert('L') if mask.mode != 'L' else mask
                
                mask_resized = mask_pil.resize(self.target_size, Image.Resampling.NEAREST)
                mask_np = (np.array(mask_resized) > 127).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_np)
                masks.append(mask_tensor)
            else:
                masks.append(None)
        
        # Stack images into batch
        images_batch = torch.stack(images).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(images_batch)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Vessel probabilities
        
        # Process results
        results = []
        for i, (prob, mask_tensor, orig_size) in enumerate(zip(probabilities, masks, original_sizes)):
            # Apply mask if available
            if mask_tensor is not None:
                mask_tensor = mask_tensor.to(self.device)
                prob = prob * mask_tensor
            
            # Resize back to original size
            prob_resized = TF.resize(
                prob.unsqueeze(0), 
                [orig_size[1], orig_size[0]]
            ).squeeze(0)
            
            prob_np = prob_resized.cpu().numpy()
            binary_mask = (prob_np > threshold).astype(np.uint8)
            
            if return_probability:
                results.append((binary_mask, prob_np))
            else:
                results.append(binary_mask)
        return results
    
    def get_model_info(self) -> Dict[str, Union[str, int, Tuple[int, int]]]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        self._ensure_model_ready()
        
        # Type assertion to help the type checker
        assert self.model is not None, "Model should be initialized"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'U-Net',
            'target_size': self.target_size,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'global_contrast_normalization': self.global_contrast_norm
        }
    
    def save_predictions(self,
                        predictions: List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
                        output_paths: List[str],
                        save_probability: bool = False):
        """
        Save predictions to files.
        
        Args:
            predictions: List of prediction arrays
            output_paths: List of output file paths
            save_probability: Whether predictions include probability maps
        """
        for pred, output_path in zip(predictions, output_paths):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if save_probability or isinstance(pred, tuple):
                binary_mask, prob_map = pred if isinstance(pred, tuple) else (pred, None)
                
                # Save binary mask
                binary_img = (binary_mask * 255).astype(np.uint8)
                Image.fromarray(binary_img).save(output_path)
                
                # Save probability map if available
                if prob_map is not None:
                    prob_path = output_path.replace('.', '_prob.')
                    prob_img = (prob_map * 255).astype(np.uint8)
                    Image.fromarray(prob_img).save(prob_path)
            else:
                # Save binary mask only
                binary_img = (pred * 255).astype(np.uint8)
                Image.fromarray(binary_img).save(output_path)

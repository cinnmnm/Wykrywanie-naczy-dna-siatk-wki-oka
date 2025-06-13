"""
Dataset Classes for Vessel Segmentation

This module contains dataset classes for loading and preprocessing
retinal images, masks, and labels for vessel segmentation training.
Implements type-safe data loading with comprehensive validation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict, Optional, Callable, Union, Any, cast
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Data.Preprocessing import ImagePreprocessing


class VesselSegmentationDataset(Dataset):
    """
    Dataset class for retinal vessel segmentation.
    
    Handles loading of image triplets (image, mask, label) and applies
    appropriate transformations including global contrast normalization.
    
    Attributes:
        data_tuples: List of (image_path, mask_path, label_path) tuples
        image_transform: Optional transform function for images
        target_size: Target size for resizing (H, W)
        global_contrast_normalization: Whether to apply GCN preprocessing
    """
    
    def __init__(self, 
                 data_tuples: List[Tuple[str, str, str]], 
                 image_transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (512, 512),
                 global_contrast_normalization: bool = False) -> None:
        """
        Initialize the dataset with type validation.
        
        Args:
            data_tuples: List of (image_path, mask_path, label_path) tuples
            image_transform: Transform to apply to images
            target_size: Target size for resizing (H, W)
            global_contrast_normalization: Whether to apply GCN preprocessing
            
        Raises:
            AssertionError: If data_tuples is empty or invalid
            FileNotFoundError: If any file paths are invalid
        """
        # Type assertions and validation
        assert isinstance(data_tuples, list), f"data_tuples must be a list, got {type(data_tuples)}"
        assert len(data_tuples) > 0, "data_tuples cannot be empty"
        assert isinstance(target_size, tuple) and len(target_size) == 2, f"target_size must be (H, W) tuple, got {target_size}"
        assert all(isinstance(x, int) and x > 0 for x in target_size), "target_size values must be positive integers"
        assert isinstance(global_contrast_normalization, bool), f"global_contrast_normalization must be bool, got {type(global_contrast_normalization)}"
        
        self.data_tuples: List[Tuple[str, str, str]] = data_tuples
        self.image_transform: Optional[Callable] = image_transform
        self.target_size: Tuple[int, int] = target_size
        self.global_contrast_normalization: bool = global_contrast_normalization
        
        # Validate all file paths exist
        self._validate_file_paths()
        
        print(f"[VesselSegmentationDataset] Initialized with {len(data_tuples)} samples")
        print(f"[VesselSegmentationDataset] Target size: {target_size}")
        print(f"[VesselSegmentationDataset] GCN enabled: {global_contrast_normalization}")
    
    def _validate_file_paths(self) -> None:
        """
        Validate that all file paths in data_tuples exist.
        
        Raises:
            FileNotFoundError: If any file path doesn't exist
            AssertionError: If tuple structure is invalid
        """
        for i, data_tuple in enumerate(self.data_tuples):
            assert isinstance(data_tuple, tuple) and len(data_tuple) == 3, \
                f"data_tuples[{i}] must be (image_path, mask_path, label_path) tuple, got {data_tuple}"
            
            image_path, mask_path, label_path = data_tuple
            for path_type, path in [("image", image_path), ("mask", mask_path), ("label", label_path)]:
                assert isinstance(path, str), f"{path_type} path must be string, got {type(path)}"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path_type} file not found: {path}")
    
    def _load_and_validate_image(self, path: str, mode: str = 'RGB') -> Image.Image:
        """
        Load and validate an image file.
        
        Args:
            path: Path to the image file
            mode: PIL image mode ('RGB' or 'L')
            
        Returns:
            Loaded PIL Image
            
        Raises:
            IOError: If image cannot be loaded
        """
        try:
            image = Image.open(path).convert(mode)
            # Validate image dimensions
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {image.size}")
            return image
        except Exception as e:
            raise IOError(f"Failed to load image from {path}: {str(e)}")
    
    def _apply_gcn_preprocessing(self, image: Image.Image) -> Image.Image:
        """
        Apply global contrast normalization to an image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            GCN processed PIL Image
        """
        image_np = np.array(image)
        image_gcn = ImagePreprocessing.apply_clahe(image_np)
        image_gcn = ImagePreprocessing.median_filter(image_gcn)
        image_gcn = ImagePreprocessing.normalize(image_gcn)
        
        # Rescale to 0-255 range for consistent processing
        min_val, max_val = image_gcn.min(), image_gcn.max()
        if max_val - min_val > 1e-8:
            image_gcn = (image_gcn - min_val) / (max_val - min_val) * 255
        else:
            image_gcn = np.zeros_like(image_gcn)
            
        image_gcn = np.clip(image_gcn, 0, 255).astype(np.uint8)
        return Image.fromarray(image_gcn)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_tuples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'image': Preprocessed image tensor [3, H, W]
                - 'label': Ground truth label tensor [1, H, W]
                - 'mask': Valid region mask tensor [1, H, W]
                - 'base_name': Base filename for reference
                
        Raises:
            IndexError: If idx is out of range
            IOError: If files cannot be loaded
        """
        # Validate index
        if not 0 <= idx < len(self.data_tuples):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data_tuples)})")
        
        image_path, mask_path, label_path = self.data_tuples[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load image, mask, and label with validation
        image = self._load_and_validate_image(image_path, 'RGB')
        mask = self._load_and_validate_image(mask_path, 'L')
        label = self._load_and_validate_image(label_path, 'L')
        
        # Apply synchronized spatial transforms (resize)
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        mask = mask.resize(self.target_size, Image.Resampling.NEAREST)
        label = label.resize(self.target_size, Image.Resampling.NEAREST)
        
        # Apply global contrast normalization if enabled
        if self.global_contrast_normalization:
            image = self._apply_gcn_preprocessing(image)
        
        # Convert to numpy arrays for processing
        mask_np = np.array(mask)
        label_np = np.array(label)
        
        # Normalize mask (valid regions = 1, invalid = 0)
        mask_binary = (mask_np > 127).astype(np.float32)
        
        # Normalize label (vessels = 1, background = 0)
        label_binary = (label_np > 127).astype(np.float32)
        
        # Apply image transforms (includes normalization)
        if self.image_transform:
            image = self.image_transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Convert mask and label to tensors
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]
        label_tensor = torch.from_numpy(label_binary).unsqueeze(0)  # [1, H, W]
        
        return {
            'image': image,
            'label': label_tensor,
            'mask': mask_tensor,
            'base_name': base_name
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a sample without loading the full data.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with sample information
        """
        if not 0 <= idx < len(self.data_tuples):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data_tuples)})")
        
        image_path, mask_path, label_path = self.data_tuples[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        return {
            'index': idx,
            'base_name': base_name,
            'image_path': image_path,
            'mask_path': mask_path,
            'label_path': label_path,
            'target_size': self.target_size,
            'gcn_enabled': self.global_contrast_normalization
        }
        if self.image_transform:
            image = self.image_transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Convert mask and label to tensors
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]
        label_tensor = torch.from_numpy(label_binary).unsqueeze(0)  # [1, H, W]
        
        return {
            'image': image,
            'label': label_tensor,
            'mask': mask_tensor,
            'base_name': base_name
        }


class SimpleImageDataset(Dataset):
    """
    Simple dataset class for inference on images without labels.
    
    Used for processing new images during inference with type safety.
    
    Attributes:
        image_paths: List of image file paths
        mask_paths: Optional list of corresponding mask paths
        image_transform: Optional transform function for images
        target_size: Target size for resizing (H, W)
        global_contrast_normalization: Whether to apply GCN preprocessing
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: Optional[List[str]] = None,
                 image_transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (512, 512),
                 global_contrast_normalization: bool = False) -> None:
        """
        Initialize simple dataset for inference with validation.
        
        Args:
            image_paths: List of image file paths
            mask_paths: Optional list of corresponding mask paths
            image_transform: Transform to apply to images
            target_size: Target size for resizing (H, W)
            global_contrast_normalization: Whether to apply GCN preprocessing
            
        Raises:
            AssertionError: If inputs are invalid
            FileNotFoundError: If image files don't exist
        """
        # Type assertions and validation
        assert isinstance(image_paths, list), f"image_paths must be a list, got {type(image_paths)}"
        assert len(image_paths) > 0, "image_paths cannot be empty"
        assert isinstance(target_size, tuple) and len(target_size) == 2, f"target_size must be (H, W) tuple, got {target_size}"
        assert all(isinstance(x, int) and x > 0 for x in target_size), "target_size values must be positive integers"
        assert isinstance(global_contrast_normalization, bool), f"global_contrast_normalization must be bool, got {type(global_contrast_normalization)}"
        
        if mask_paths is not None:
            assert isinstance(mask_paths, list), f"mask_paths must be a list or None, got {type(mask_paths)}"
            assert len(mask_paths) == len(image_paths), f"mask_paths length ({len(mask_paths)}) must match image_paths length ({len(image_paths)})"        
        self.image_paths: List[str] = image_paths
        self.mask_paths: List[Optional[str]] = cast(List[Optional[str]], mask_paths) if mask_paths is not None else [None] * len(image_paths)
        self.image_transform: Optional[Callable] = image_transform
        self.target_size: Tuple[int, int] = target_size
        self.global_contrast_normalization: bool = global_contrast_normalization
        
        # Validate all image paths exist
        self._validate_image_paths()
        
        print(f"[SimpleImageDataset] Initialized with {len(image_paths)} images")
        print(f"[SimpleImageDataset] Target size: {target_size}")
        print(f"[SimpleImageDataset] GCN enabled: {global_contrast_normalization}")
        
    def _validate_image_paths(self) -> None:
        """
        Validate that all image paths exist.
        
        Raises:
            FileNotFoundError: If any image path doesn't exist
            AssertionError: If path types are invalid
        """
        for i, image_path in enumerate(self.image_paths):
            assert isinstance(image_path, str), f"image_paths[{i}] must be string, got {type(image_path)}"
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Validate mask paths if provided
        for i, mask_path in enumerate(self.mask_paths):
            if mask_path is not None:
                assert isinstance(mask_path, str), f"mask_paths[{i}] must be string or None, got {type(mask_path)}"
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file not found (will be ignored): {mask_path}")
    
    def _load_and_validate_image(self, path: str, mode: str = 'RGB') -> Image.Image:
        """
        Load and validate an image file.
        
        Args:
            path: Path to the image file
            mode: PIL image mode ('RGB' or 'L')
            
        Returns:
            Loaded PIL Image
            
        Raises:
            IOError: If image cannot be loaded
        """
        try:
            image = Image.open(path).convert(mode)
            # Validate image dimensions
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {image.size}")
            return image
        except Exception as e:
            raise IOError(f"Failed to load image from {path}: {str(e)}")
    
    def _apply_gcn_preprocessing(self, image: Image.Image) -> Image.Image:
        """
        Apply global contrast normalization to an image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            GCN processed PIL Image
        """
        image_np = np.array(image)
        image_gcn = ImagePreprocessing.global_contrast_normalization(image_np)
        
        # Rescale to 0-255 range for consistent processing
        min_val, max_val = image_gcn.min(), image_gcn.max()
        if max_val - min_val > 1e-8:
            image_gcn = (image_gcn - min_val) / (max_val - min_val) * 255
        else:
            image_gcn = np.zeros_like(image_gcn)
            
        image_gcn = np.clip(image_gcn, 0, 255).astype(np.uint8)
        return Image.fromarray(image_gcn)
        
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Tuple[int, int], None]]:
        """
        Get a single image sample for inference.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'image': Preprocessed image tensor [3, H, W]
                - 'mask': Optional mask tensor [1, H, W] or None
                - 'base_name': Base filename for reference
                - 'original_size': Original image size (W, H)
                
        Raises:
            IndexError: If idx is out of range
            IOError: If image cannot be loaded
        """
        # Validate index
        if not 0 <= idx < len(self.image_paths):
            raise IndexError(f"Index {idx} out of range [0, {len(self.image_paths)})")
        
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load and process image
        image = self._load_and_validate_image(image_path, 'RGB')
        original_size = image.size  # (W, H)
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Apply global contrast normalization if enabled
        if self.global_contrast_normalization:
            image = self._apply_gcn_preprocessing(image)
        
        # Load mask if available
        mask_tensor: Optional[torch.Tensor] = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = self._load_and_validate_image(mask_path, 'L')
                mask = mask.resize(self.target_size, Image.Resampling.NEAREST)
                mask_np = (np.array(mask) > 127).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
            except Exception as e:
                print(f"Warning: Failed to load mask {mask_path}: {str(e)}")
                mask_tensor = None
        
        # Apply image transforms
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'mask': mask_tensor,
            'base_name': base_name,
            'original_size': original_size
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get information about a sample without loading the full data.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with sample information
        """
        if not 0 <= idx < len(self.image_paths):
            raise IndexError(f"Index {idx} out of range [0, {len(self.image_paths)})")
        
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        return {
            'index': idx,
            'base_name': base_name,
            'image_path': image_path,
            'mask_path': mask_path,
            'target_size': self.target_size,
            'gcn_enabled': self.global_contrast_normalization,
            'has_mask': mask_path is not None and os.path.exists(mask_path) if mask_path else False
        }

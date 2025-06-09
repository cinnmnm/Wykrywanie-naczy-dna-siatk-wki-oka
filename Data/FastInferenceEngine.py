import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import time
import concurrent.futures

class FastInferenceEngine:
    """
    Optimized inference engine for patch-based segmentation.
    Minimizes GPU-CPU transfers and maximizes parallelization.
    """
    
    def __init__(self, model, device='cuda', patch_size=27, batch_size=1024):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.half = patch_size // 2
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
    
    def predict_full_image(self, gpu_dataset, img_idx: int, mask_threshold: float = 0.5, 
                          overlap_strategy: str = 'center') -> torch.Tensor:
        """
        Fast full image segmentation using GPU-optimized patch extraction and batched inference.
        
        Args:
            gpu_dataset: GPUMappedDataset instance with images/masks/labels on GPU
            img_idx: Index of image to segment
            mask_threshold: Threshold for valid patches (patches must have mask > threshold)
            overlap_strategy: How to handle overlapping predictions ('center', 'average', 'voting')
        
        Returns:
            torch.Tensor: Prediction map of shape [H, W] on GPU
        """
        
        # Get image and mask tensors (already on GPU)
        image = gpu_dataset.images[img_idx]  # [C, H, W]
        mask = gpu_dataset.masks[img_idx]    # [1, H, W] or [H, W]
        
        if mask.ndim == 3:
            mask = mask[0]  # [H, W]
        
        H, W = mask.shape
        
        # Create output prediction map
        pred_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        if overlap_strategy == 'average':
            # For averaging, we need to track counts and sums
            vote_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
            count_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        # Fast valid patch discovery using unfold (vectorized)
        valid_coords = self._find_valid_patch_centers_fast(mask, mask_threshold)
        
        if len(valid_coords) == 0:
            print(f"No valid patches found for image {img_idx}")
            return pred_map
        
        print(f"Found {len(valid_coords)} valid patches for inference")
        
        # Extract all patches in batches and run inference
        num_patches = len(valid_coords)
        
        for start_idx in range(0, num_patches, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_patches)
            batch_coords = valid_coords[start_idx:end_idx]
            
            # Extract batch of patches (all on GPU)
            batch_patches = self._extract_patch_batch(image, batch_coords)
            
            # Run inference on batch
            with torch.no_grad():
                outputs = self.model(batch_patches)
                predictions = torch.argmax(outputs, dim=1)  # [batch_size]
            
            # Update prediction map based on strategy
            if overlap_strategy == 'center':
                # Only predict at center pixel of each patch
                for i, (y, x) in enumerate(batch_coords):
                    pred_map[y, x] = predictions[i]
            
            elif overlap_strategy == 'average':
                # Average predictions over overlapping patches
                for i, (y, x) in enumerate(batch_coords):
                    # Get patch region
                    y_start, y_end = y - self.half, y + self.half + 1
                    x_start, x_end = x - self.half, x + self.half + 1
                    
                    # Add to vote map and count map
                    vote_map[y_start:y_end, x_start:x_end] += predictions[i].float()
                    count_map[y_start:y_end, x_start:x_end] += 1
        
        if overlap_strategy == 'average':
            # Finalize averaged predictions
            valid_mask = count_map > 0
            pred_map[valid_mask] = (vote_map[valid_mask] / count_map[valid_mask]).round().long()
        
        return pred_map
    
    def _find_valid_patch_centers_fast(self, mask: torch.Tensor, threshold: float) -> List[Tuple[int, int]]:
        """
        Fast vectorized patch validity checking using unfold.
        
        Args:
            mask: 2D mask tensor [H, W] on GPU
            threshold: Minimum threshold for patch validity
        
        Returns:
            List of (y, x) coordinates for valid patch centers
        """
        H, W = mask.shape
        
        if H < self.patch_size or W < self.patch_size:
            return []
        
        # Use unfold to extract all patches at once
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        patches = F.unfold(mask_4d, kernel_size=self.patch_size, stride=1)  # [1, patch_size^2, N]
        
        # Check validity: all pixels in patch must be > threshold
        valid = (patches > threshold).all(dim=1).squeeze(0)  # [N]
        
        # Convert flat indices to (y, x) coordinates
        out_h = H - self.patch_size + 1
        out_w = W - self.patch_size + 1
        
        # Get coordinates of valid patches
        valid_flat_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)  # [num_valid]
        
        # Convert to (y, x) and add half patch size to get center coordinates
        y_coords = (valid_flat_indices // out_w) + self.half
        x_coords = (valid_flat_indices % out_w) + self.half
        
        # Convert to list of tuples (on CPU)
        coords = list(zip(y_coords.cpu().numpy(), x_coords.cpu().numpy()))
        
        return coords
    
    def _extract_patch_batch(self, image: torch.Tensor, coords: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Extract a batch of patches from an image efficiently.
        
        Args:
            image: Image tensor [C, H, W] on GPU
            coords: List of (y, x) center coordinates
        
        Returns:
            torch.Tensor: Batch of patches [batch_size, C, patch_size, patch_size]
        """
        batch_size = len(coords)
        C, H, W = image.shape
        
        # Pre-allocate batch tensor
        batch = torch.zeros((batch_size, C, self.patch_size, self.patch_size), 
                           dtype=image.dtype, device=self.device)
        
        # Extract each patch
        for i, (y, x) in enumerate(coords):
            y_start, y_end = y - self.half, y + self.half + 1
            x_start, x_end = x - self.half, x + self.half + 1
            
            batch[i] = image[:, y_start:y_end, x_start:x_end]
        
        return batch
    
    def predict_multiple_images(self, gpu_dataset, img_indices: List[int], 
                               mask_threshold: float = 0.5) -> List[torch.Tensor]:
        """
        Predict segmentation for multiple images efficiently.
        
        Args:
            gpu_dataset: GPUMappedDataset instance
            img_indices: List of image indices to process
            mask_threshold: Threshold for valid patches
        
        Returns:
            List of prediction maps (torch.Tensor)
        """
        results = []
        
        for img_idx in img_indices:
            print(f"Processing image {img_idx}...")
            start_time = time.time()
            
            pred_map = self.predict_full_image(
                gpu_dataset, img_idx, mask_threshold=mask_threshold
            )
            
            end_time = time.time()
            print(f"Image {img_idx} processed in {end_time - start_time:.2f}s")
            
            results.append(pred_map)
        
        return results

class SlidingWindowInference:
    """
    Alternative inference method using sliding window (no mask-based filtering).
    Useful when you want dense predictions over the entire image.
    """
    
    def __init__(self, model, device='cuda', patch_size=27, stride=None, batch_size=1024):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2  # Default to 50% overlap
        self.batch_size = batch_size
        self.half = patch_size // 2
        
        self.model.to(device)
        self.model.eval()
    
    def predict_sliding_window(self, gpu_dataset, img_idx: int) -> torch.Tensor:
        """
        Dense sliding window inference over entire image.
        
        Args:
            gpu_dataset: GPUMappedDataset instance
            img_idx: Index of image to segment
        
        Returns:
            torch.Tensor: Dense prediction map [H, W]
        """
        image = gpu_dataset.images[img_idx]  # [C, H, W]
        C, H, W = image.shape
        
        # Create prediction and count maps
        pred_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        count_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        # Generate all sliding window positions
        coords = []
        for y in range(self.half, H - self.half, self.stride):
            for x in range(self.half, W - self.half, self.stride):
                coords.append((y, x))
        
        print(f"Sliding window: {len(coords)} patches")
        
        # Process in batches
        for start_idx in range(0, len(coords), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(coords))
            batch_coords = coords[start_idx:end_idx]
            
            # Extract patches
            batch_patches = self._extract_patch_batch(image, batch_coords)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_patches)
                predictions = torch.argmax(outputs, dim=1).float()
            
            # Accumulate predictions
            for i, (y, x) in enumerate(batch_coords):
                y_start, y_end = y - self.half, y + self.half + 1
                x_start, x_end = x - self.half, x + self.half + 1
                
                pred_map[y_start:y_end, x_start:x_end] += predictions[i]
                count_map[y_start:y_end, x_start:x_end] += 1
        
        # Average overlapping predictions
        valid_mask = count_map > 0
        final_pred = torch.zeros_like(pred_map, dtype=torch.long)
        final_pred[valid_mask] = (pred_map[valid_mask] / count_map[valid_mask]).round().long()
        
        return final_pred
    
    def _extract_patch_batch(self, image: torch.Tensor, coords: List[Tuple[int, int]]) -> torch.Tensor:
        """Extract batch of patches from image."""
        batch_size = len(coords)
        C, H, W = image.shape
        
        batch = torch.zeros((batch_size, C, self.patch_size, self.patch_size), 
                           dtype=image.dtype, device=self.device)
        
        for i, (y, x) in enumerate(coords):
            y_start, y_end = y - self.half, y + self.half + 1
            x_start, x_end = x - self.half, x + self.half + 1
            
            batch[i] = image[:, y_start:y_end, x_start:x_end]
        
        return batch

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union
import time
from scipy.ndimage import convolve

class ImprovedInferenceEngine:
    """
    Improved inference engine that matches original patch discovery logic
    and provides better border handling for dense segmentation.
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
    
    def predict_full_image_original_logic(self, gpu_dataset, img_idx: int, 
                                        overlap_strategy: str = 'center') -> torch.Tensor:
        """
        Fast full image segmentation using EXACT original patch discovery logic.
        This matches the convolution-based validation from PatchFeatureExtractor.
        
        Args:
            gpu_dataset: GPUMappedDataset instance with images/masks/labels on GPU
            img_idx: Index of image to segment
            overlap_strategy: How to handle overlapping predictions ('center', 'average')
        
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
            vote_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
            count_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        # Use ORIGINAL patch discovery logic (convolution-based)
        valid_coords = self._find_valid_patch_centers_original_logic(mask)
        
        if len(valid_coords) == 0:
            print(f"No valid patches found for image {img_idx}")
            return pred_map
        
        print(f"Found {len(valid_coords)} valid patches for inference (original logic)")
        
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
    
    def _find_valid_patch_centers_original_logic(self, mask: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Exact replication of original patch discovery logic using convolution.
        This matches PatchFeatureExtractor.extract_patches() validation.
        
        Args:
            mask: 2D mask tensor [H, W] on GPU
        
        Returns:
            List of (y, x) coordinates for valid patch centers
        """
        H, W = mask.shape
        
        if H < self.patch_size or W < self.patch_size:
            return []
        
        # Convert to CPU numpy for convolution (matches original)
        mask_np = mask.cpu().numpy()
        
        # Convert to binary mask (matches original logic)
        mask_gray = (mask_np > 0).astype(np.uint8)
        
        # Create convolution kernel (all ones)
        kernel = np.ones((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # Convolve mask with kernel
        valid_mask = convolve(mask_gray, kernel, mode='constant', cval=0)
        
        # Valid patches are those where ALL pixels are within mask
        valid_mask = valid_mask == (self.patch_size * self.patch_size)
        
        print(f"Original logic: {np.sum(valid_mask)} valid patch centers")
        
        # Find coordinates of valid patch centers
        coords = []
        half_patch = self.patch_size // 2
        
        for y in range(half_patch, H - half_patch):
            for x in range(half_patch, W - half_patch):
                if valid_mask[y, x]:
                    coords.append((y, x))
        
        return coords
    
    def _find_valid_patch_centers_relaxed(self, mask: torch.Tensor, 
                                        min_valid_fraction: float = 0.8) -> List[Tuple[int, int]]:
        """
        Relaxed patch discovery that allows patches with partial mask coverage.
        Better for border handling.
        
        Args:
            mask: 2D mask tensor [H, W] on GPU
            min_valid_fraction: Minimum fraction of patch that must be valid (0.0 to 1.0)
        
        Returns:
            List of (y, x) coordinates for valid patch centers
        """
        H, W = mask.shape
        
        if H < self.patch_size or W < self.patch_size:
            return []
        
        # Use unfold to extract all patches at once
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        patches = F.unfold(mask_4d, kernel_size=self.patch_size, stride=1)  # [1, patch_size^2, N]
        
        # Count valid pixels per patch
        valid_pixels_per_patch = (patches > 0).sum(dim=1).squeeze(0).float()  # [N]
        total_pixels = self.patch_size * self.patch_size
        
        # Check if fraction of valid pixels meets threshold
        valid_fraction = valid_pixels_per_patch / total_pixels
        valid = valid_fraction >= min_valid_fraction
        
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
        
        print(f"Relaxed logic ({min_valid_fraction*100:.1f}% threshold): {len(coords)} valid patch centers")
        
        return coords
    
    def predict_full_image_relaxed(self, gpu_dataset, img_idx: int, 
                                 min_valid_fraction: float = 0.8,
                                 overlap_strategy: str = 'center') -> torch.Tensor:
        """
        Full image segmentation with relaxed patch validation for better border coverage.
        
        Args:
            gpu_dataset: GPUMappedDataset instance
            img_idx: Index of image to segment
            min_valid_fraction: Minimum fraction of patch that must be valid
            overlap_strategy: How to handle overlapping predictions
        
        Returns:
            torch.Tensor: Prediction map of shape [H, W] on GPU
        """
        
        # Get image and mask tensors
        image = gpu_dataset.images[img_idx]  # [C, H, W]
        mask = gpu_dataset.masks[img_idx]    # [1, H, W] or [H, W]
        
        if mask.ndim == 3:
            mask = mask[0]  # [H, W]
        
        H, W = mask.shape
        
        # Create output prediction map
        pred_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        if overlap_strategy == 'average':
            vote_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
            count_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        # Use relaxed patch discovery
        valid_coords = self._find_valid_patch_centers_relaxed(mask, min_valid_fraction)
        
        if len(valid_coords) == 0:
            print(f"No valid patches found for image {img_idx}")
            return pred_map
        
        # Extract all patches in batches and run inference
        num_patches = len(valid_coords)
        
        for start_idx in range(0, num_patches, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_patches)
            batch_coords = valid_coords[start_idx:end_idx]
            
            # Extract batch of patches
            batch_patches = self._extract_patch_batch_safe(image, mask, batch_coords)
            
            # Run inference on batch
            with torch.no_grad():
                outputs = self.model(batch_patches)
                predictions = torch.argmax(outputs, dim=1)  # [batch_size]
            
            # Update prediction map based on strategy
            if overlap_strategy == 'center':
                # Only predict at center pixel if it's within mask
                for i, (y, x) in enumerate(batch_coords):
                    if mask[y, x] > 0:  # Only predict within mask
                        pred_map[y, x] = predictions[i]
            
            elif overlap_strategy == 'average':
                # Average predictions over overlapping patches (mask-aware)
                for i, (y, x) in enumerate(batch_coords):
                    # Get patch region
                    y_start, y_end = y - self.half, y + self.half + 1
                    x_start, x_end = x - self.half, x + self.half + 1
                    
                    # Only accumulate within mask
                    patch_mask = mask[y_start:y_end, x_start:x_end] > 0
                    vote_map[y_start:y_end, x_start:x_end] += (predictions[i].float() * patch_mask.float())
                    count_map[y_start:y_end, x_start:x_end] += patch_mask.long()
        
        if overlap_strategy == 'average':
            # Finalize averaged predictions
            valid_mask = count_map > 0
            pred_map[valid_mask] = (vote_map[valid_mask] / count_map[valid_mask]).round().long()
        
        return pred_map
    
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
    
    def _extract_patch_batch_safe(self, image: torch.Tensor, mask: torch.Tensor, 
                                coords: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Extract patches with boundary checking and padding if needed.
        
        Args:
            image: Image tensor [C, H, W] on GPU
            mask: Mask tensor [H, W] on GPU
            coords: List of (y, x) center coordinates
        
        Returns:
            torch.Tensor: Batch of patches [batch_size, C, patch_size, patch_size]
        """
        batch_size = len(coords)
        C, H, W = image.shape
        
        # Pre-allocate batch tensor
        batch = torch.zeros((batch_size, C, self.patch_size, self.patch_size), 
                           dtype=image.dtype, device=self.device)
        
        # Extract each patch with boundary checking
        for i, (y, x) in enumerate(coords):
            y_start = max(0, y - self.half)
            y_end = min(H, y + self.half + 1)
            x_start = max(0, x - self.half)
            x_end = min(W, x + self.half + 1)
            
            # Calculate offsets for placing in patch
            patch_y_start = self.half - (y - y_start)
            patch_y_end = patch_y_start + (y_end - y_start)
            patch_x_start = self.half - (x - x_start)
            patch_x_end = patch_x_start + (x_end - x_start)
            
            # Extract and place patch
            batch[i, :, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
                image[:, y_start:y_end, x_start:x_end]
        
        return batch


class DenseInferenceEngine:
    """
    Alternative dense segmentation approaches beyond patch-based methods.
    """
    
    def __init__(self, model, device='cuda', patch_size=27):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.half = patch_size // 2
        
        self.model.to(device)
        self.model.eval()
    
    def predict_sliding_window_dense(self, gpu_dataset, img_idx: int, 
                                   stride: int = 1, batch_size: int = 2048) -> torch.Tensor:
        """
        Ultra-dense sliding window inference with stride=1 for complete coverage.
        
        Args:
            gpu_dataset: GPUMappedDataset instance
            img_idx: Index of image to segment
            stride: Stride for sliding window (1 = every pixel)
            batch_size: Number of patches to process at once
        
        Returns:
            torch.Tensor: Dense prediction map [H, W]
        """
        image = gpu_dataset.images[img_idx]  # [C, H, W]
        mask = gpu_dataset.masks[img_idx]    # [1, H, W] or [H, W]
        
        if mask.ndim == 3:
            mask = mask[0]  # [H, W]
        
        C, H, W = image.shape
        
        # Create prediction and count maps
        pred_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        count_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        # Generate all possible patch centers within valid region
        coords = []
        for y in range(self.half, H - self.half, stride):
            for x in range(self.half, W - self.half, stride):
                if mask[y, x] > 0:  # Only process centers within mask
                    coords.append((y, x))
        
        print(f"Dense sliding window: {len(coords)} patches (stride={stride})")
        
        # Process in batches
        for start_idx in range(0, len(coords), batch_size):
            end_idx = min(start_idx + batch_size, len(coords))
            batch_coords = coords[start_idx:end_idx]
            
            # Extract patches
            batch_patches = self._extract_patch_batch(image, batch_coords)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_patches)
                predictions = torch.argmax(outputs, dim=1).float()
            
            # Accumulate predictions with mask-aware averaging
            for i, (y, x) in enumerate(batch_coords):
                y_start, y_end = y - self.half, y + self.half + 1
                x_start, x_end = x - self.half, x + self.half + 1
                
                # Get mask for this patch region
                patch_mask = mask[y_start:y_end, x_start:x_end] > 0
                
                # Accumulate only within mask
                pred_map[y_start:y_end, x_start:x_end] += (predictions[i] * patch_mask.float())
                count_map[y_start:y_end, x_start:x_end] += patch_mask.long()
        
        # Average overlapping predictions
        valid_mask = count_map > 0
        final_pred = torch.zeros_like(pred_map, dtype=torch.long)
        final_pred[valid_mask] = (pred_map[valid_mask] / count_map[valid_mask]).round().long()
        
        return final_pred
    
    def predict_adaptive_patches(self, gpu_dataset, img_idx: int, 
                               base_stride: int = 8, adaptive_zones: bool = True) -> torch.Tensor:
        """
        Adaptive patch sampling with higher density near vessel structures.
        
        Args:
            gpu_dataset: GPUMappedDataset instance
            img_idx: Index of image to segment
            base_stride: Base stride for patch sampling
            adaptive_zones: Whether to use adaptive sampling based on image features
        
        Returns:
            torch.Tensor: Prediction map [H, W]
        """
        image = gpu_dataset.images[img_idx]  # [C, H, W]
        mask = gpu_dataset.masks[img_idx]    # [1, H, W] or [H, W]
        
        if mask.ndim == 3:
            mask = mask[0]  # [H, W]
        
        C, H, W = image.shape
        
        # Create prediction and count maps
        pred_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        count_map = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        coords = []
        
        if adaptive_zones:
            # Compute image gradients to identify high-detail regions
            gray_image = image.mean(dim=0)  # Simple grayscale conversion
            
            # Compute gradient magnitude
            grad_x = torch.abs(gray_image[1:, :] - gray_image[:-1, :])
            grad_y = torch.abs(gray_image[:, 1:] - gray_image[:, :-1])
            
            # Pad gradients to match original size
            grad_x = F.pad(grad_x, (0, 0, 0, 1), mode='replicate')
            grad_y = F.pad(grad_y, (0, 1, 0, 0), mode='replicate')
            
            gradient_magnitude = grad_x + grad_y
            
            # Adaptive stride based on gradient
            for y in range(self.half, H - self.half, base_stride):
                for x in range(self.half, W - self.half, base_stride):
                    if mask[y, x] > 0:
                        # Check gradient in local region
                        local_grad = gradient_magnitude[y-self.half:y+self.half+1, 
                                                      x-self.half:x+self.half+1].mean()
                        
                        # Add base coordinate
                        coords.append((y, x))
                        
                        # Add additional samples in high-gradient regions
                        if local_grad > gradient_magnitude.mean():
                            # Higher density sampling
                            for dy in range(-base_stride//2, base_stride//2+1, base_stride//4):
                                for dx in range(-base_stride//2, base_stride//2+1, base_stride//4):
                                    new_y, new_x = y + dy, x + dx
                                    if (self.half <= new_y < H - self.half and 
                                        self.half <= new_x < W - self.half and 
                                        mask[new_y, new_x] > 0):
                                        coords.append((new_y, new_x))
        else:
            # Regular grid sampling
            for y in range(self.half, H - self.half, base_stride):
                for x in range(self.half, W - self.half, base_stride):
                    if mask[y, x] > 0:
                        coords.append((y, x))
        
        print(f"Adaptive patches: {len(coords)} patches")
        
        # Process patches in batches
        batch_size = 1024
        for start_idx in range(0, len(coords), batch_size):
            end_idx = min(start_idx + batch_size, len(coords))
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
                
                patch_mask = mask[y_start:y_end, x_start:x_end] > 0
                pred_map[y_start:y_end, x_start:x_end] += (predictions[i] * patch_mask.float())
                count_map[y_start:y_end, x_start:x_end] += patch_mask.long()
        
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


class PatchDiscoveryComparison:
    """
    Utility class to compare different patch discovery methods and understand discrepancies.
    """
    
    def __init__(self, patch_size=27):
        self.patch_size = patch_size
        self.half = patch_size // 2
    
    def compare_methods(self, mask: torch.Tensor, threshold: float = 0.5) -> dict:
        """
        Compare all patch discovery methods to understand discrepancies.
        
        Args:
            mask: Mask tensor [H, W]
            threshold: Threshold for fast method
        
        Returns:
            dict: Results from all methods
        """
        results = {}
        
        # Method 1: Original convolution-based (exact replication)
        coords_original = self._original_convolution_method(mask)
        results['original_convolution'] = {
            'count': len(coords_original),
            'coords': coords_original[:100],  # First 100 for inspection
            'description': 'Exact replication of PatchFeatureExtractor logic'
        }
        
        # Method 2: Fast unfold-based (your current implementation)
        coords_fast = self._fast_unfold_method(mask, threshold)
        results['fast_unfold'] = {
            'count': len(coords_fast),
            'coords': coords_fast[:100],
            'description': f'Fast unfold with threshold {threshold}'
        }
        
        # Method 3: Relaxed validation (partial mask coverage)
        coords_relaxed = self._relaxed_method(mask, min_fraction=0.8)
        results['relaxed_80'] = {
            'count': len(coords_relaxed),
            'coords': coords_relaxed[:100],
            'description': 'Relaxed validation (80% mask coverage)'
        }
        
        # Method 4: Very relaxed validation
        coords_very_relaxed = self._relaxed_method(mask, min_fraction=0.5)
        results['relaxed_50'] = {
            'count': len(coords_very_relaxed),
            'coords': coords_very_relaxed[:100],
            'description': 'Very relaxed validation (50% mask coverage)'
        }
        
        # Print comparison
        print("\n=== PATCH DISCOVERY COMPARISON ===")
        for method, data in results.items():
            print(f"{method:20s}: {data['count']:8d} patches - {data['description']}")
        
        # Find overlaps
        set_original = set(coords_original)
        set_fast = set(coords_fast)
        overlap = len(set_original & set_fast)
        print(f"\nOverlap between original and fast: {overlap} patches")
        print(f"Original only: {len(set_original - set_fast)} patches")
        print(f"Fast only: {len(set_fast - set_original)} patches")
        
        return results
    
    def _original_convolution_method(self, mask: torch.Tensor) -> List[Tuple[int, int]]:
        """Exact replication of original convolution-based method."""
        mask_np = mask.cpu().numpy()
        mask_gray = (mask_np > 0).astype(np.uint8)
        
        kernel = np.ones((self.patch_size, self.patch_size), dtype=np.uint8)
        valid_mask = convolve(mask_gray, kernel, mode='constant', cval=0)
        valid_mask = valid_mask == (self.patch_size * self.patch_size)
        
        coords = []
        H, W = mask.shape
        for y in range(self.half, H - self.half):
            for x in range(self.half, W - self.half):
                if valid_mask[y, x]:
                    coords.append((y, x))
        
        return coords
    
    def _fast_unfold_method(self, mask: torch.Tensor, threshold: float) -> List[Tuple[int, int]]:
        """Fast unfold-based method."""
        H, W = mask.shape
        if H < self.patch_size or W < self.patch_size:
            return []
        
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        patches = F.unfold(mask_4d, kernel_size=self.patch_size, stride=1)
        
        valid = (patches > threshold).all(dim=1).squeeze(0)
        
        out_h = H - self.patch_size + 1
        out_w = W - self.patch_size + 1
        
        valid_flat_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        y_coords = (valid_flat_indices // out_w) + self.half
        x_coords = (valid_flat_indices % out_w) + self.half
        
        return list(zip(y_coords.cpu().numpy(), x_coords.cpu().numpy()))
    
    def _relaxed_method(self, mask: torch.Tensor, min_fraction: float) -> List[Tuple[int, int]]:
        """Relaxed method allowing partial mask coverage."""
        H, W = mask.shape
        if H < self.patch_size or W < self.patch_size:
            return []
        
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        patches = F.unfold(mask_4d, kernel_size=self.patch_size, stride=1)
        
        valid_pixels_per_patch = (patches > 0).sum(dim=1).squeeze(0).float()
        total_pixels = self.patch_size * self.patch_size
        valid_fraction = valid_pixels_per_patch / total_pixels
        valid = valid_fraction >= min_fraction
        
        out_h = H - self.patch_size + 1
        out_w = W - self.patch_size + 1
        
        valid_flat_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        y_coords = (valid_flat_indices // out_w) + self.half
        x_coords = (valid_flat_indices % out_w) + self.half
        
        return list(zip(y_coords.cpu().numpy(), x_coords.cpu().numpy()))

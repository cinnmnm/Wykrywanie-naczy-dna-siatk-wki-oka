"""
Patch Feature Extractor for Decision Forest Model

This module provides feature extraction capabilities for patches
using the proven logic from PatchFeatureExtractor.py.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from skimage.measure import moments_central


class PatchFeatureExtractor:
    """
    Feature extractor using the proven methods from Data/PatchFeatureExtractor.py
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        print(f"[PatchFeatureExtractor] Initialized with config: {self.config}")
    
    def extract_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract features from a patch using the proven methods.
        Uses color_variance, central_moments, and hu_moments like the working code.
        """
        color_vars = self.color_variance(patch)      # c features
        central_moms = self.central_moments(patch)   # 3*(c+1) features  
        hu_moms = self.hu_moments(patch)             # 7*(c+1) features

        features = np.concatenate([color_vars, central_moms, hu_moms])
        return features
    
    def color_variance(self, patch: np.ndarray) -> np.ndarray:
        """
        Calculate color variance for each channel.
        patch: h x w x c
        Output: c (variance for each channel)
        """
        if patch.ndim != 3:
            raise ValueError(f"Patch should be of shape: h x w x c. Got {patch.ndim} dimensions instead.")
        return np.var(patch, axis=(0, 1))

    def central_moments(self, patch: np.ndarray) -> np.ndarray:
        """
        Calculate central moments for each channel + grayscale.
        patch: h x w x c
        Output: 3 values for each channel + 3 for gray: [m[2,0], m[1,1], m[0,2]] * (c+1)
        """
        if patch.ndim != 3:
            raise ValueError(f"Patch should be of shape: h x w x c. Got {patch.ndim} dimensions instead.")
        features = []
        
        # For each color channel
        for c in range(patch.shape[2]):
            m = moments_central(patch[..., c])
            features.extend([m[2, 0], m[1, 1], m[0, 2]])
        
        # For grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        m_gray = moments_central(gray)
        features.extend([m_gray[2, 0], m_gray[1, 1], m_gray[0, 2]])
        
        return np.array(features)

    def hu_moments(self, patch: np.ndarray) -> np.ndarray:
        """
        Calculate Hu moments for each channel + grayscale.
        patch: h x w x c
        Output: 7 values for each channel + 7 for gray: 7 * (c+1)
        """
        if patch.ndim != 3:
            raise ValueError("Input must be 3D (h x w x c)")
        features = []
        
        # For each color channel
        for c in range(patch.shape[2]):
            m = cv2.moments(patch[..., c])
            hu = cv2.HuMoments(m).flatten()
            features.extend(hu)
        
        # For grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        m_gray = cv2.moments(gray)
        hu_gray = cv2.HuMoments(m_gray).flatten()
        features.extend(hu_gray)
        
        return np.array(features)
    
    def get_feature_count(self, patch_shape: Tuple[int, int, int]) -> int:
        """
        Calculate the expected number of features for a patch shape.
        """
        h, w, c = patch_shape
        color_vars = c                    # c features
        central_moms = 3 * (c + 1)       # 3 for each channel + 3 for gray
        hu_moms = 7 * (c + 1)            # 7 for each channel + 7 for gray
        return color_vars + central_moms + hu_moms
    
    def get_feature_names(self, patch_shape: Tuple[int, int, int]) -> List[str]:
        """
        Generate feature names for a given patch shape.
        """
        h, w, c = patch_shape
        names = []
        
        # Color variance features
        for ch in range(c):
            names.append(f'color_var_ch{ch}')
        
        # Central moments features
        for ch in range(c):
            names.extend([f'central_m20_ch{ch}', f'central_m11_ch{ch}', f'central_m02_ch{ch}'])
        names.extend(['central_m20_gray', 'central_m11_gray', 'central_m02_gray'])
        
        # Hu moments features
        for ch in range(c):
            for i in range(7):
                names.append(f'hu_{i}_ch{ch}')
        for i in range(7):
            names.append(f'hu_{i}_gray')
        
        return names

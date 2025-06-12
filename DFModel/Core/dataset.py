"""
Dataset Classes for Decision Forest Model

This module provides dataset classes for loading and preprocessing
retinal vessel images for patch-based decision forest classification.
Uses the proven patch extraction logic from PatchDatasetGenerator.py.
"""

import numpy as np
import cv2
import os
from typing import List, Tuple, Optional, Dict, Any
from .feature_extractor import PatchFeatureExtractor


class ImageDataset:
    """
    Loads and serves (image, mask, label) triplets as numpy arrays.
    """
    def __init__(self, image_tuples: List[Tuple[str, str, str, str]], transform_fn: Optional[Any] = None):
        self.image_tuples = image_tuples
        self.transform_fn = transform_fn
        self.images = []
        self.masks = []
        self.labels = []
        self._load_all()

    def _load_all(self):
        for _, img_path, mask_path, label_path in self.image_tuples:
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if self.transform_fn:
                img, mask, label = self.transform_fn(img, mask, label)
            self.images.append(img)
            self.masks.append(mask)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.labels[idx]


class PatchDataset:
    """
    Extracts valid patches and features using the proven logic from PatchDatasetGenerator.py.
    """
    def __init__(self, image_dataset: ImageDataset, patch_size: int = 27, 
                 feature_extractor: Optional[PatchFeatureExtractor] = None, 
                 samples_per_image: int = 1000, positive_ratio: float = 0.5):
        self.image_dataset = image_dataset
        self.patch_size = patch_size
        self.feature_extractor = feature_extractor or PatchFeatureExtractor()
        self.samples_per_image = samples_per_image
        self.positive_ratio = positive_ratio
        self.patches = None
        self.patch_labels = None
        self.features = None
        self._extract_patches_and_features()

    def _extract_patches_and_features(self):
        """
        Extract patches using the proven logic from PatchDatasetGenerator.py
        """
        n = self.patch_size
        half_patch = n // 2
        patches = []
        patch_labels = []
        
        print(f"[PatchDataset] Extracting patches from {len(self.image_dataset)} images...")
        
        for img_idx in range(len(self.image_dataset)):
            img, mask, label = self.image_dataset[img_idx]
            h, w = label.shape[:2]
            
            # Find all valid patch centers for this image
            pos_idx = []
            neg_idx = []
            
            for center_y in range(half_patch, h - half_patch):
                for center_x in range(half_patch, w - half_patch):
                    # Extract mask patch using the same logic as PatchDatasetGenerator
                    patch_x_start = center_x - half_patch
                    patch_x_end = center_x + half_patch + 1
                    patch_y_start = center_y - half_patch
                    patch_y_end = center_y + half_patch + 1
                    
                    mask_patch = mask[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
                    
                    # Check if all mask pixels are valid (using np.all like in working code)
                    if np.all(mask_patch):
                        # Get label for center pixel
                        patch_label_val = label[center_y, center_x]
                        
                        # Normalize if needed (like in PatchDatasetGenerator)
                        if hasattr(patch_label_val, 'max') and patch_label_val.max() > 1:
                            patch_label_val = patch_label_val / 255.0
                        
                        patch_label_binary = int(patch_label_val > 0.5)
                        
                        if patch_label_binary == 1:
                            pos_idx.append((img_idx, center_y, center_x))
                        else:
                            neg_idx.append((img_idx, center_y, center_x))
            
            print(f"  Image {img_idx}: found {len(pos_idx)} positive, {len(neg_idx)} negative patch centers")
            
            # Sample patches according to desired ratio
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            
            n_pos = int(self.samples_per_image * self.positive_ratio)
            n_neg = self.samples_per_image - n_pos
            
            selected_pos = pos_idx[:min(len(pos_idx), n_pos)]
            selected_neg = neg_idx[:min(len(neg_idx), n_neg)]
            
            # Extract patches
            for img_idx, center_y, center_x in selected_pos:
                patch = img[center_y - half_patch:center_y + half_patch + 1, 
                           center_x - half_patch:center_x + half_patch + 1]
                patches.append(patch)
                patch_labels.append(1)
            
            for img_idx, center_y, center_x in selected_neg:
                patch = img[center_y - half_patch:center_y + half_patch + 1, 
                           center_x - half_patch:center_x + half_patch + 1]
                patches.append(patch)
                patch_labels.append(0)
            
            print(f"  Selected {len(selected_pos)} positive, {len(selected_neg)} negative patches")
        
        self.patches = np.array(patches) if patches else np.array([])
        self.patch_labels = np.array(patch_labels) if patch_labels else np.array([])
        
        print(f"[PatchDataset] Total patches extracted: {len(self.patches)}")
        
        # Extract features from patches
        if len(self.patches) > 0:
            print(f"[PatchDataset] Extracting features from {len(self.patches)} patches...")
            features = []
            for i, patch in enumerate(self.patches):
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(self.patches)} patches")
                features.append(self.feature_extractor.extract_features(patch))
            self.features = np.array(features)
            print(f"[PatchDataset] Feature extraction completed. Shape: {self.features.shape}")
        else:
            self.features = np.array([])

    def get_features_and_labels(self):
        return self.features, self.patch_labels
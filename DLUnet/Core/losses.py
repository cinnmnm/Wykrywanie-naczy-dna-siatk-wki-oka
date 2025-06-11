"""
Loss Functions for Vessel Segmentation

This module contains specialized loss functions for handling class imbalance
and improving performance on vessel segmentation tasks.
"""

import torch
import torch.nn as nn


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for handling class imbalance in vessel segmentation.
    
    Combines class weights with focal loss to address both class imbalance
    and hard negative mining.
    """
    
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        """
        Initialize Weighted Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1)
            gamma: Focusing parameter (default: 2)
            weight: Class weights tensor
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets, mask=None):
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, 1, H, W] or [B, H, W]
            mask: Valid region mask [B, 1, H, W] or [B, H, W]
        
        Returns:
            Computed loss value
        """
        ce_loss = nn.functional.cross_entropy(
            inputs, targets.long().squeeze(1), 
            weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if mask is not None:
            # Only compute loss on valid regions
            mask = mask.squeeze(1)  # Remove channel dimension if present
            focal_loss = focal_loss * mask
            
            if self.reduction == 'mean':
                return focal_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal_loss.sum()
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    The Dice coefficient measures overlap between prediction and ground truth,
    making it well-suited for segmentation where we care about region overlap.
    """
    
    def __init__(self, smooth=1e-6):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets, mask=None):
        """
        Compute Dice loss.
        
        Args:
            inputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, 1, H, W] or [B, H, W]
            mask: Valid region mask [B, 1, H, W] or [B, H, W]
        
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        # Convert to probabilities
        probs = torch.softmax(inputs, dim=1)[:, 1]  # Vessel class probability
        targets = targets.squeeze(1).float()
        
        if mask is not None:
            mask = mask.squeeze(1).float()
            probs = probs * mask
            targets = targets * mask
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss function using both Focal Loss and Dice Loss.
    
    This combination helps with both class imbalance (Focal) and 
    shape accuracy (Dice).
    """
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, **kwargs):
        """
        Initialize Combined Loss.
        
        Args:
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            **kwargs: Arguments passed to individual loss functions
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = WeightedFocalLoss(**kwargs)
        self.dice_loss = DiceLoss()
    
    def forward(self, inputs, targets, mask=None):
        """Compute combined loss"""
        focal = self.focal_loss(inputs, targets, mask)
        dice = self.dice_loss(inputs, targets, mask)
        
        return self.focal_weight * focal + self.dice_weight * dice

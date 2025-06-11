"""
U-Net Model Architecture for Retinal Vessel Segmentation

This module contains the U-Net implementation used for vessel segmentation.
The architecture includes skip connections and is optimized for biomedical image segmentation.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from typing import Dict, Tuple, Union


class UNetSegmentation(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    The U-Net consists of a contracting path (encoder) and an expanding path (decoder).
    Skip connections between corresponding encoder and decoder layers help preserve
    spatial information lost during downsampling.
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_features: int = 64):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (2 for vessel/background)
            base_features: Base number of features in first layer
        """
        super(UNetSegmentation, self).__init__()
        
        # Store configuration for later reference
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_features = base_features
        
        # Encoder (Contracting Path)
        self.enc1 = self._double_conv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(base_features * 8, base_features * 16)
        
        # Decoder (Expanding Path)
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, 2)
        self.dec4 = self._double_conv(base_features * 16, base_features * 8)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, 2)
        self.dec3 = self._double_conv(base_features * 8, base_features * 4)
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, 2)
        self.dec2 = self._double_conv(base_features * 4, base_features * 2)
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, 2)
        self.dec1 = self._double_conv(base_features * 2, base_features)
          # Output
        self.out_conv = nn.Conv2d(base_features, num_classes, 1)
    
    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            
        Returns:
            Sequential module containing the double convolution block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, self._match_size(e4, d4)], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, self._match_size(e3, d3)], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, self._match_size(e2, d2)], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, self._match_size(e1, d1)], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)
    
    def _match_size(self, skip: torch.Tensor, upsampled: torch.Tensor) -> torch.Tensor:
        """
        Ensure skip connection and upsampled feature maps have matching sizes
        
        Args:
            skip: Skip connection tensor from encoder
            upsampled: Upsampled tensor from decoder
            
        Returns:
            Resized skip tensor matching upsampled tensor size
        """        
        if skip.shape[2:] != upsampled.shape[2:]:
            skip = TF.resize(skip, list(upsampled.shape[2:]))
        return skip
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Get information about the model architecture
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'U-Net',
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'base_features': self.base_features,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming 32-bit floats
        }
    
    def get_feature_maps_info(self) -> Dict[str, Union[Dict[str, int], int]]:
        """
        Get information about feature map sizes at each level
        
        Returns:
            Dictionary with feature map information
        """
        return {
            'encoder': {
                'enc1': self.base_features,
                'enc2': self.base_features * 2,
                'enc3': self.base_features * 4,
                'enc4': self.base_features * 8,
            },
            'bottleneck': self.base_features * 16,
            'decoder': {
                'dec4': self.base_features * 8,
                'dec3': self.base_features * 4,
                'dec2': self.base_features * 2,
                'dec1': self.base_features,
            },
            'output': self.num_classes
        }
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """
        Validate input tensor dimensions and properties
        
        Args:
            x: Input tensor to validate
            
        Raises:
            ValueError: If input tensor is invalid
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (batch, channels, height, width), got {x.dim()}D")
        
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        
        # Check if spatial dimensions are divisible by 16 (due to 4 pooling operations)
        if x.shape[2] % 16 != 0 or x.shape[3] % 16 != 0:
            print(f"Warning: Input spatial dimensions ({x.shape[2]}, {x.shape[3]}) are not divisible by 16. "
                  "This may cause size mismatches in skip connections.")
    
    def forward_with_validation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input validation
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        self._validate_input(x)
        return self.forward(x)
    
    def print_model_summary(self) -> None:
        """Print a summary of the model architecture"""
        info = self.get_model_info()
        feature_info = self.get_feature_maps_info()
        
        print("=" * 60)
        print(f"U-Net Model Summary")
        print("=" * 60)
        print(f"Model Name: {info['model_name']}")
        print(f"Input Channels: {info['in_channels']}")
        print(f"Output Classes: {info['num_classes']}")
        print(f"Base Features: {info['base_features']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print("-" * 60)
        print("Feature Map Channels:")
        print(f"  Encoder: {feature_info['encoder']}")
        print(f"  Bottleneck: {feature_info['bottleneck']}")        
        print(f"  Decoder: {feature_info['decoder']}")
        print(f"  Output: {feature_info['output']}")
        print("=" * 60)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder weights for transfer learning"""
        encoder_modules = [self.enc1, self.enc2, self.enc3, self.enc4, self.bottleneck]
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = False
        print("Encoder layers frozen for transfer learning")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("All model parameters unfrozen")
    
    def get_trainable_params_info(self) -> Dict[str, int]:
        """Get detailed information about trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }

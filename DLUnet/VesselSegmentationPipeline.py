"""
Legacy Vessel Segmentation Pipeline (DEPRECATED)

This file is now deprecated in favor of the new modular architecture.
Please use the new pipeline from DLUnet.pipeline instead:

    from DLUnet import VesselSegmentationPipeline
    
    pipeline = VesselSegmentationPipeline(config_path="config.yaml")
    pipeline.prepare_data()
    pipeline.train()

This file is kept for backward compatibility with existing notebooks.
"""

import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms.functional as TF
import cv2
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import warnings

# Local imports
from Data.DatasetSupplier import DatasetSupplier
from Data.Preprocessing import ImagePreprocessing
from Util.config import Config

warnings.filterwarnings("ignore", category=UserWarning)

class VesselSegmentationDataset(Dataset):
    """
    Dataset class for retinal vessel segmentation.
    
    Handles loading of image triplets (image, mask, manual_label) with
    configurable preprocessing and augmentations.    """
    
    def __init__(self, 
                 data_tuples: List[Tuple[str, str, str, str]], 
                 image_transform=None, 
                 mask_transform=None,
                 augmentation_transform=None,
                 target_size: Tuple[int, int] = (512, 512),
                 global_contrast_normalization: bool = False):
        """
        Args:
            data_tuples: List of (base_name, image_path, manual_path, mask_path)
            image_transform: Transform applied to input images
            mask_transform: Transform applied to masks (validity regions)
            augmentation_transform: Joint transform applied to both image and label
            target_size: Target size for resizing (H, W)
            global_contrast_normalization: Whether to apply global contrast normalization
        """
        self.data_tuples = data_tuples
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augmentation_transform = augmentation_transform
        self.target_size = target_size
        self.global_contrast_normalization = global_contrast_normalization
        
        print(f"[VesselSegmentationDataset] Initialized with {len(data_tuples)} samples")
        print(f"[VesselSegmentationDataset] Target size: {target_size}")
        print(f"[VesselSegmentationDataset] Global contrast normalization: {global_contrast_normalization}")
    
    def __len__(self):
        return len(self.data_tuples)
    
    def __getitem__(self, idx):
        base_name, image_path, manual_path, mask_path = self.data_tuples[idx]
        try:
            # Load images as PIL
            image = Image.open(image_path).convert('RGB')
            manual_label = Image.open(manual_path).convert('L')
            mask = Image.open(mask_path).convert('L')

            # --- JOINT SPATIAL TRANSFORMS ---
            # Apply ALL spatial transforms identically to image, mask, and label
            
            # 1. Resize (always applied)
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            manual_label = manual_label.resize(self.target_size, Image.Resampling.NEAREST)
            mask = mask.resize(self.target_size, Image.Resampling.NEAREST)

            # 2. Data augmentation (if enabled)
            if self.augmentation_transform:
                # Set random seed for this sample to ensure identical transforms
                seed = np.random.randint(0, 2**32)                # Random horizontal flip
                np.random.seed(seed)
                if np.random.rand() < 0.5:
                    image = ImageOps.mirror(image)
                    manual_label = ImageOps.mirror(manual_label)
                    mask = ImageOps.mirror(mask)
                
                # Random vertical flip
                np.random.seed(seed + 1)
                if np.random.rand() < 0.5:
                    image = ImageOps.flip(image)
                    manual_label = ImageOps.flip(manual_label)
                    mask = ImageOps.flip(mask)
                
                # Random rotation (90, 180, 270 degrees)
                np.random.seed(seed + 2)
                angle = np.random.choice([0, 90, 180, 270])
                if angle != 0:
                    image = image.rotate(angle, expand=False)
                    manual_label = manual_label.rotate(angle, expand=False)
                    mask = mask.rotate(angle, expand=False)

            # Convert to numpy arrays
            image_np = np.array(image)
            manual_label_np = np.array(manual_label)
            mask_np = np.array(mask)

            # --- IMAGE-ONLY TRANSFORMS ---
            # Apply global contrast normalization if enabled (ONLY to image)
            if self.global_contrast_normalization:
                image_np = ImagePreprocessing.global_contrast_normalization(image_np)
                image_np = np.clip((image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8) * 255, 0, 255).astype(np.uint8)

            # Convert image back to PIL for torchvision transforms
            image = Image.fromarray(image_np)

            # Apply image-specific transforms (normalization, etc.)
            if self.image_transform:
                image = self.image_transform(image)
            else:
                # Default transforms: ToTensor + Normalize
                image = T.ToTensor()(image)
                image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

            # --- LABEL/MASK PROCESSING ---
            # Convert to binary and then to tensors
            manual_label_binary = (manual_label_np > 127).astype(np.float32)
            mask_binary = (mask_np > 127).astype(np.float32)
            
            # Convert to tensors with channel dimension
            manual_label_tensor = torch.from_numpy(manual_label_binary).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

            return {
                'image': image,
                'label': manual_label_tensor,
                'mask': mask_tensor,
                'base_name': base_name
            }
            
        except Exception as e:
            print(f"Error loading sample {idx} ({base_name}): {e}")
            return {
                'image': torch.zeros(3, *self.target_size),
                'label': torch.zeros(1, *self.target_size),
                'mask': torch.zeros(1, *self.target_size),
                'base_name': f"error_{idx}"
            }

class UNetSegmentation(nn.Module):
    """
    U-Net architecture for retinal vessel segmentation.
    
    A flexible U-Net implementation with configurable depth and feature channels.
    """
    
    def __init__(self, in_channels=3, num_classes=2, base_features=64):
        super().__init__()
        
        # Encoder
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
        
        # Decoder
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
        
    def _double_conv(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
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
    
    def _match_size(self, skip, upsampled):
        """Ensure skip connection and upsampled feature maps have matching sizes"""
        if skip.shape[2:] != upsampled.shape[2:]:
            skip = TF.resize(skip, upsampled.shape[2:])
        return skip

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for handling class imbalance in vessel segmentation.
    
    Combines class weights with focal loss to address both class imbalance
    and hard negative mining.
    """
    
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets, mask=None):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets.long().squeeze(1), 
            weight=self.weight, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if mask is not None:
            # Only compute loss on valid regions
            mask = mask.squeeze(1)  # Remove channel dimension
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

class VesselSegmentationPipeline:
    """
    Main pipeline class for retinal vessel segmentation.
    
    Encapsulates the entire workflow from data loading to model training and inference.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the segmentation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = Config.load(config_path)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None
        
        # Training state
        self.best_val_score = 0.0
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
        
        print(f"[VesselSegmentationPipeline] Initialized with device: {self.device}")
        print(f"[VesselSegmentationPipeline] Config loaded from: {config_path}")
    
    def prepare_data(self, 
                    train_split: float = 0.7, 
                    val_split: float = 0.15, 
                    test_split: float = 0.15,
                    target_size: Tuple[int, int] = None):
        """
        Prepare datasets and data loaders.
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            target_size: Target image size (H, W)
        """
        print("[VesselSegmentationPipeline] Preparing data...")
        
        if target_size is None:
            target_size = tuple(self.config.get('resize_shape', [512, 512]))
        
        # Get dataset tuples
        data_tuples = DatasetSupplier.get_dataset(self.config_path)
        print(f"Found {len(data_tuples)} complete data samples")
        
        if len(data_tuples) == 0:
            raise ValueError("No valid data samples found!")
          # Create transforms
        train_transform = self._create_train_transforms()
        val_transform = self._create_val_transforms()
        
        # Check if global contrast normalization is enabled
        global_contrast_norm = self.config.get('preprocessing', {}).get('global_contrast_normalization', False)
        
        # Create full dataset
        full_dataset = VesselSegmentationDataset(
            data_tuples=data_tuples,
            image_transform=train_transform,
            target_size=target_size,
            global_contrast_normalization=global_contrast_norm
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42))
        )
        
        # Update transforms for validation and test sets
        # Create new datasets with validation transforms
        val_dataset_new = VesselSegmentationDataset(
            data_tuples=[full_dataset.data_tuples[i] for i in val_dataset.indices],
            image_transform=val_transform,
            target_size=target_size,
            global_contrast_normalization=global_contrast_norm
        )
        test_dataset_new = VesselSegmentationDataset(
            data_tuples=[full_dataset.data_tuples[i] for i in test_dataset.indices],
            image_transform=val_transform,
            target_size=target_size,
            global_contrast_normalization=global_contrast_norm
        )
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 4)
        num_workers = self.config.get('num_workers', 0)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset_new, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset_new, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        print(f"[VesselSegmentationPipeline] Data preparation complete")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
    
    def _create_train_transforms(self):
        """Create training transforms with preprocessing and augmentation"""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #T.RandomHorizontalFlip(p=0.5),
            #T.RandomVerticalFlip(p=0.5),
            #T.RandomRotation(degrees=10),
        ])
    
    def _create_val_transforms(self):
        """Create validation/test transforms (no augmentation)"""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def initialize_model(self, 
                        in_channels: int = 3, 
                        num_classes: int = 2, 
                        base_features: int = 64):
        """
        Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            base_features: Base number of features in the first layer
        """
        print("[VesselSegmentationPipeline] Initializing model...")
        
        self.model = UNetSegmentation(
            in_channels=in_channels,
            num_classes=num_classes,
            base_features=base_features
        ).to(self.device)
        
        # Initialize optimizer
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Initialize loss function with class weights
        class_weights = self.config.get('class_weights', [1.0, 1.0])
        if isinstance(class_weights, list):
            class_weights = torch.tensor(class_weights, device=self.device)
        
        self.criterion = WeightedFocalLoss(
            alpha=1.0,
            gamma=2.0,
            weight=class_weights
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"[VesselSegmentationPipeline] Model initialized")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Learning rate: {lr}")
        print(f"  Class weights: {class_weights}")
    
    def train(self, num_epochs: int = None, save_path: str = None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_path: Path to save the best model
        """
        if self.model is None:
            self.initialize_model()
        
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 50)
        
        if save_path is None:
            save_path = self.config.get('model_save_path', 'DLUnet/SavedModels/best_model.pth')
        
        print(f"[VesselSegmentationPipeline] Starting training for {num_epochs} epochs")
        
        # Create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_iou'].append(val_metrics['iou'])
            
            # Save best model
            if val_metrics['iou'] > self.best_val_score:
                self.best_val_score = val_metrics['iou']
                torch.save(self.model.state_dict(), save_path)
                print(f"  New best model saved! IoU: {val_metrics['iou']:.4f}")
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
        
        print(f"[VesselSegmentationPipeline] Training complete!")
        print(f"  Best validation IoU: {self.best_val_score:.4f}")
        print(f"  Model saved to: {save_path}")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels, masks)
                
                # Get predictions
                predictions = torch.softmax(outputs, dim=1)[:, 1] > 0.5  # Vessel class
                
                # Mask out invalid regions
                valid_mask = masks.squeeze(1).bool()
                predictions_masked = predictions[valid_mask]
                labels_masked = labels.squeeze(1)[valid_mask]
                
                all_predictions.append(predictions_masked.cpu().numpy())
                all_labels.append(labels_masked.cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        iou = jaccard_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        metrics = {
            'iou': iou,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        return total_loss / num_batches, metrics
    
    def evaluate(self, loader=None):
        """
        Evaluate the model on test set.
        
        Args:
            loader: DataLoader to evaluate on (defaults to test_loader)
        """
        if loader is None:
            loader = self.test_loader
        
        if loader is None:
            raise ValueError("No test loader available. Call prepare_data() first.")
        
        print("[VesselSegmentationPipeline] Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                predictions = torch.softmax(outputs, dim=1)[:, 1] > 0.5
                
                # Mask out invalid regions
                valid_mask = masks.squeeze(1).bool()
                predictions_masked = predictions[valid_mask]
                labels_masked = labels.squeeze(1)[valid_mask]
                
                all_predictions.append(predictions_masked.cpu().numpy())
                all_labels.append(labels_masked.cpu().numpy())
        
        # Calculate comprehensive metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        metrics = {
            'accuracy': (all_predictions == all_labels).mean(),
            'iou': jaccard_score(all_labels, all_predictions),
            'f1': f1_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions)
        }
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
      def predict(self, image_paths: List[str], output_dir: str = None, threshold: float = 0.5):
        """
        Run inference on new images.
        
        Args:
            image_paths: List of paths to images
            output_dir: Directory to save prediction results
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train model or load weights first.")
        
        print(f"[VesselSegmentationPipeline] Predicting on {len(image_paths)} images...")
        
        self.model.eval()
        predictions = []
        
        transform = self._create_val_transforms()
        target_size = tuple(self.config.get('resize_shape', [512, 512]))
        
        with torch.no_grad():
            for img_path in image_paths:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                original_size = image.size  # (W, H)
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                # Run inference
                output = self.model(image_tensor)
                prediction = torch.softmax(output, dim=1)[0, 1]  # Vessel probability
                
                # Try to find corresponding mask file
                mask_path = self._find_mask_path(img_path)
                if mask_path and os.path.exists(mask_path):
                    # Load and process mask
                    mask = Image.open(mask_path).convert('L')
                    mask = mask.resize(target_size, Image.Resampling.NEAREST)
                    mask_tensor = torch.from_numpy(np.array(mask) > 127).float().to(self.device)
                    
                    # Apply mask: set predictions outside valid region to 0
                    prediction = prediction * mask_tensor
                
                # Convert back to original size
                prediction_resized = TF.resize(
                    prediction.unsqueeze(0), 
                    [original_size[1], original_size[0]]  # PIL uses (W,H), torch uses (H,W)
                ).squeeze(0)
                
                prediction_np = prediction_resized.cpu().numpy()
                predictions.append(prediction_np)
                
                # Save if output directory specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    # Save probability map
                    prob_path = os.path.join(output_dir, f"{base_name}_prob.png")
                    prob_img = (prediction_np * 255).astype(np.uint8)
                    Image.fromarray(prob_img).save(prob_path)
                    
                    # Save binary mask
                    binary_path = os.path.join(output_dir, f"{base_name}_mask.png")
                    binary_img = ((prediction_np > threshold) * 255).astype(np.uint8)
                    Image.fromarray(binary_img).save(binary_path)        
        print(f"[VesselSegmentationPipeline] Prediction complete!")
        if output_dir:
            print(f"  Results saved to: {output_dir}")
        
        return predictions
    
    def _find_mask_path(self, image_path: str) -> str:
        """
        Find corresponding mask file for an image path.
        Assumes mask files are in a 'mask' directory with '_mask' suffix.
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path)
        
        # Try different possible mask locations
        possible_paths = [
            # Same directory with _mask suffix
            os.path.join(image_dir, f"{base_name}_mask.tif"),
            os.path.join(image_dir, f"{base_name}_mask.png"),
            # Mask subdirectory
            os.path.join(os.path.dirname(image_dir), "mask", f"{base_name}_mask.tif"),
            os.path.join(os.path.dirname(image_dir), "mask", f"{base_name}_mask.png"),
            # Adjacent mask directory
            os.path.join(image_dir, "..", "mask", f"{base_name}_mask.tif"),
            os.path.join(image_dir, "..", "mask", f"{base_name}_mask.png"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_weights(self, path: str):
        """Load model weights from file"""
        if self.model is None:
            self.initialize_model()
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[VesselSegmentationPipeline] Loaded weights from: {path}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history['train_loss']:
            print("No training history available")
            return
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # IoU plot
        ax2.plot(epochs, self.training_history['val_iou'], 'g-', label='Val IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.set_title('Validation IoU')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = VesselSegmentationPipeline("config.yaml")
    
    # Prepare data
    pipeline.prepare_data(train_split=0.7, val_split=0.15, test_split=0.15)
    
    # Initialize model
    pipeline.initialize_model(base_features=32)  # Smaller model for testing
    
    # Train
    pipeline.train(num_epochs=5)
    
    # Evaluate
    pipeline.evaluate()
    
    # Plot training history
    pipeline.plot_training_history()

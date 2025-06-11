"""
Training Pipeline for Vessel Segmentation

This module contains the training logic and pipeline management
for retinal vessel segmentation models.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from typing import Dict, List, Optional, Tuple, Union
import random

from .model import UNetSegmentation
from .losses import WeightedFocalLoss, DiceLoss, CombinedLoss
from .dataset import VesselSegmentationDataset
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Data.DatasetSupplier import DatasetSupplier
from Util.config import Config


class VesselSegmentationTrainer:
    """
    Training pipeline for vessel segmentation models.
    
    Handles model initialization, training loop, validation, and checkpointing.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = Config.load(config_path)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components with proper type hints
        self.model: Optional[UNetSegmentation] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        
        # Training state
        self.best_val_score: float = 0.0
        self.training_history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'val_iou': []}
        
        print(f"[VesselSegmentationTrainer] Initialized with device: {self.device}")
        print(f"[VesselSegmentationTrainer] Config loaded from: {config_path}")
    
    def prepare_data(self, 
                    train_split: float = 0.7, 
                    val_split: float = 0.15, 
                    test_split: float = 0.15,
                    target_size: Optional[Tuple[int, int]] = None):
        """
        Prepare datasets and data loaders.
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            target_size: Target image size (H, W)
        """
        print("[VesselSegmentationTrainer] Preparing data...")
        
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
        
        # Split data tuples first (more efficient and cleaner)
        total_size = len(data_tuples)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        random.seed(self.config.get('seed', 42))
        shuffled_tuples = data_tuples.copy()
        random.shuffle(shuffled_tuples)
        
        train_tuples = shuffled_tuples[:train_size]
        val_tuples = shuffled_tuples[train_size:train_size + val_size]
        test_tuples = shuffled_tuples[train_size + val_size:]
        
        # Create separate datasets with appropriate transforms
        train_dataset = VesselSegmentationDataset(
            data_tuples=train_tuples,
            image_transform=train_transform,  # Training transforms (with augmentation)
            target_size=target_size,
            global_contrast_normalization=global_contrast_norm
        )
        
        val_dataset_new = VesselSegmentationDataset(
            data_tuples=val_tuples,
            image_transform=val_transform,   # Validation transforms (no augmentation)
            target_size=target_size,
            global_contrast_normalization=global_contrast_norm
        )
        
        test_dataset_new = VesselSegmentationDataset(
            data_tuples=test_tuples,
            image_transform=val_transform,   # Test transforms (no augmentation)
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
        
        print(f"[VesselSegmentationTrainer] Data preparation complete")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
    
    def _create_train_transforms(self):
        """Create training transforms with preprocessing and augmentation"""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        print("[VesselSegmentationTrainer] Initializing model...")
        
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
        loss_type = self.config.get('loss_type', 'focal')
        class_weights = self.config.get('class_weights', [1.0, 1.0])
        
        if isinstance(class_weights, list):
            class_weights = torch.tensor(class_weights, device=self.device)
        
        if loss_type == 'focal':
            self.criterion = WeightedFocalLoss(
                alpha=1,
                gamma=2,
                weight=class_weights
            )
        elif loss_type == 'dice':
            self.criterion = DiceLoss()
        elif loss_type == 'combined':
            self.criterion = CombinedLoss(
                focal_weight=0.5,
                dice_weight=0.5,
                weight=class_weights
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"[VesselSegmentationTrainer] Model initialized")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Learning rate: {lr}")
        print(f"  Loss type: {loss_type}")
        print(f"  Class weights: {class_weights}")    
    def train(self, num_epochs: int = 50, save_path: Optional[str] = None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_path: Path to save the best model
        """
        # Ensure model is initialized
        if self.model is None:
            self.initialize_model()
        
        # Ensure all components are ready for training
        self._ensure_initialized()
        
        if save_path is None:
            save_path = self.config.get('model_save_path', 'DLUnet/SavedModels/best_model.pth')
        
        print(f"[VesselSegmentationTrainer] Starting training for {num_epochs} epochs")
        
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
        
        print(f"[VesselSegmentationTrainer] Training complete!")
        print(f"  Best validation IoU: {self.best_val_score:.4f}")
        print(f"  Model saved to: {save_path}")
    
    def _train_epoch(self) -> float:
        """Train for one epoch"""
        # All components are guaranteed to be initialized when this is called
        self.model.train()  # type: ignore
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:  # type: ignore
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()  # type: ignore
            
            outputs = self.model(images)  # type: ignore
            loss = self.criterion(outputs, labels, masks)  # type: ignore
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # type: ignore
            self.optimizer.step()  # type: ignore
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        # Type assertions to help the type checker
        assert self.model is not None, "Model must be initialized before validation"
        assert self.criterion is not None, "Criterion must be initialized before validation"
        assert self.val_loader is not None, "Validation loader must be prepared before validation"
        
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
        
        iou = float(jaccard_score(all_labels, all_predictions))
        f1 = float(f1_score(all_labels, all_predictions))
        precision = float(precision_score(all_labels, all_predictions))
        recall = float(recall_score(all_labels, all_predictions))
        
        metrics = {
            'iou': iou,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        return total_loss / num_batches, metrics
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on test set.
        
        Args:
            loader: DataLoader to evaluate on (defaults to test_loader)
        """
        if loader is None:
            loader = self.test_loader
        
        if loader is None:
            raise ValueError("No test loader available. Call prepare_data() first.")
        
        # Type assertions to help the type checker
        assert self.model is not None, "Model must be initialized before evaluation"
        
        print("[VesselSegmentationTrainer] Evaluating model...")
        
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
            'accuracy': float((all_predictions == all_labels).mean()),
            'iou': float(jaccard_score(all_labels, all_predictions)),
            'f1': float(f1_score(all_labels, all_predictions)),
            'precision': float(precision_score(all_labels, all_predictions)),
            'recall': float(recall_score(all_labels, all_predictions))
        }
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def load_weights(self, path: str) -> None:
        """Load model weights from file"""
        if self.model is None:
            self.initialize_model()
        
        # Type assertion to help the type checker
        assert self.model is not None, "Model should be initialized"
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[VesselSegmentationTrainer] Loaded weights from: {path}")
    
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
    
    def _ensure_initialized(self) -> None:
        """Ensure all required components are initialized for training/evaluation"""
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model, optimizer, and criterion must be initialized. Call initialize_model() first.")
        
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("Data loaders must be prepared. Call prepare_data() first.")
    
    def _ensure_model_initialized(self) -> None:
        """Ensure model is initialized for evaluation/loading weights"""
        if self.model is None:
            raise RuntimeError("Model must be initialized. Call initialize_model() first.")

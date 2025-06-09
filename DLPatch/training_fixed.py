#!/usr/bin/env python3
"""
Optimized Training Script for Retinal Vessel Segmentation
=========================================================

This training script leverages the fast inference techniques developed in FastInferenceEngine.py
to achieve maximum training performance while avoiding precomputing patch indices.

Key optimizations:
1. Vectorized patch discovery using unfold operations
2. Dynamic balanced patch sampling per epoch
3. GPU-optimized data pipeline
4. Memory-efficient training loop
5. Fast tensor transforms without PIL conversions

Usage:
    python training.py
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
from typing import List, Tuple, Optional, Dict, Any
import gc
from contextlib import contextmanager

# Import existing modules
from Util.config import Config
from Util.Evaluate import Evaluate
from Data.Preprocessing import ImagePreprocessing
from Data.GPUMappedDataset import GPUMappedDataset
from Data.DatasetSupplier import DatasetSupplier
from DLPatch.DLModel import DLModel, gcn_pil_to_tensor_transform


class FastPatchSampler(Dataset):
    """
    Fast patch sampler that discovers patches dynamically using vectorized operations.
    This avoids precomputing all patch indices and enables efficient sampling.
    """
    
    def __init__(self, 
                 gpu_dataset: GPUMappedDataset,
                 image_indices: List[int],
                 patch_size: int = 27,
                 mask_threshold: float = 0.5,
                 samples_per_epoch: int = 10000,
                 class_balance_ratio: float = 1.0,
                 transform=None,
                 device: str = 'cuda'):
        """
        Initialize the fast patch sampler.
        
        Args:
            gpu_dataset: GPUMappedDataset with images, masks, labels on GPU
            image_indices: List of image indices to sample from
            patch_size: Size of patches to extract
            mask_threshold: Minimum mask value for valid patches
            samples_per_epoch: Number of patches to sample per epoch
            class_balance_ratio: Ratio of class 0 to class 1 samples
            transform: Transform function to apply to patches
            device: Device to use for computations
        """
        self.gpu_dataset = gpu_dataset
        self.image_indices = image_indices
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.mask_threshold = mask_threshold
        self.samples_per_epoch = samples_per_epoch
        self.class_balance_ratio = class_balance_ratio
        self.transform = transform
        self.device = device
        
        # Cache valid patch centers for each image to avoid recomputation
        self._valid_centers_cache = {}
        self._discover_valid_patches()
        
        # Generate balanced samples for current epoch
        self._generate_epoch_samples()
    
    def _find_valid_patch_centers_vectorized(self, img_idx: int) -> torch.Tensor:
        """
        Fast vectorized patch center discovery using unfold operations.
        Based on FastInferenceEngine._find_valid_patch_centers_fast.
        """
        if img_idx in self._valid_centers_cache:
            return self._valid_centers_cache[img_idx]
        
        mask = self.gpu_dataset.masks[img_idx]  # [C, H, W] or [H, W]
        
        # Ensure 2D mask
        if mask.ndim == 3:
            mask = mask[0]  # Take first channel
        elif mask.ndim != 2:
            raise ValueError(f"Invalid mask dimensions: {mask.shape}")
        
        H, W = mask.shape
        if H < self.patch_size or W < self.patch_size:
            self._valid_centers_cache[img_idx] = torch.empty((0, 2), device=self.device, dtype=torch.long)
            return self._valid_centers_cache[img_idx]
        
        # Add batch and channel dimensions for unfold
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Use unfold to extract all possible patches
        patches = torch.nn.functional.unfold(
            mask_4d, 
            kernel_size=self.patch_size, 
            stride=1
        )  # [1, patch_size*patch_size, N]
        
        # Check which patches are entirely within the mask
        valid_mask = (patches > self.mask_threshold).all(dim=1).squeeze(0)  # [N]
        
        # Generate center coordinates
        out_h = H - self.patch_size + 1
        out_w = W - self.patch_size + 1
        y_coords, x_coords = torch.meshgrid(
            torch.arange(out_h, device=self.device),
            torch.arange(out_w, device=self.device),
            indexing='ij'
        )
        
        # Convert to patch centers (add half patch size)
        y_centers = y_coords.flatten() + self.half
        x_centers = x_coords.flatten() + self.half
        
        # Filter valid centers
        valid_centers = torch.stack([y_centers[valid_mask], x_centers[valid_mask]], dim=1)
        
        self._valid_centers_cache[img_idx] = valid_centers
        return valid_centers
    
    def _discover_valid_patches(self):
        """Discover all valid patch centers for assigned images."""
        print(f"Discovering valid patches for {len(self.image_indices)} images...")
        
        self.image_valid_centers = {}
        self.image_class_centers = {}  # Separate by class for faster sampling
        
        total_patches = 0
        for img_idx in self.image_indices:
            centers = self._find_valid_patch_centers_vectorized(img_idx)
            
            if len(centers) == 0:
                self.image_valid_centers[img_idx] = centers
                self.image_class_centers[img_idx] = {
                    0: torch.empty((0, 2), device=self.device, dtype=torch.long),
                    1: torch.empty((0, 2), device=self.device, dtype=torch.long)
                }
                continue
            
            # Get labels for all centers at once
            labels = self.gpu_dataset.labels[img_idx, :, centers[:, 0], centers[:, 1]]
            if labels.ndim > 1:
                labels = labels[0]  # Take first channel if multi-channel
            
            # Convert to class labels
            class_labels = (labels > 0.5).long()
            
            # Separate by class
            class_0_mask = class_labels == 0
            class_1_mask = class_labels == 1
            
            self.image_valid_centers[img_idx] = centers
            self.image_class_centers[img_idx] = {
                0: centers[class_0_mask],
                1: centers[class_1_mask]
            }
            
            total_patches += len(centers)
        
        print(f"Found {total_patches} total valid patches across {len(self.image_indices)} images")
    
    def _generate_epoch_samples(self):
        """Generate balanced sample indices for current epoch."""
        # Calculate target samples per class
        if self.class_balance_ratio == 1.0:
            samples_per_class = self.samples_per_epoch // 2
            target_class_0 = samples_per_class
            target_class_1 = samples_per_class
        else:
            total_ratio = 1.0 + self.class_balance_ratio
            target_class_1 = int(self.samples_per_epoch / total_ratio)
            target_class_0 = int(target_class_1 * self.class_balance_ratio)
        
        # Collect all available patches by class
        all_class_0_patches = []
        all_class_1_patches = []
        
        for img_idx in self.image_indices:
            class_centers = self.image_class_centers[img_idx]
            
            # Add image index to each center coordinate
            for center in class_centers[0]:
                all_class_0_patches.append((img_idx, center[0].item(), center[1].item()))
            
            for center in class_centers[1]:
                all_class_1_patches.append((img_idx, center[0].item(), center[1].item()))
        
        # Sample balanced patches
        self.current_samples = []
        
        if all_class_0_patches and target_class_0 > 0:
            if len(all_class_0_patches) >= target_class_0:
                sampled_0 = np.random.choice(len(all_class_0_patches), target_class_0, replace=False)
                self.current_samples.extend([all_class_0_patches[i] for i in sampled_0])
            else:
                # If not enough patches, use all available
                self.current_samples.extend(all_class_0_patches)
        
        if all_class_1_patches and target_class_1 > 0:
            if len(all_class_1_patches) >= target_class_1:
                sampled_1 = np.random.choice(len(all_class_1_patches), target_class_1, replace=False)
                self.current_samples.extend([all_class_1_patches[i] for i in sampled_1])
            else:
                # If not enough patches, use all available
                self.current_samples.extend(all_class_1_patches)
        
        # Shuffle samples
        np.random.shuffle(self.current_samples)
        
        print(f"Generated {len(self.current_samples)} balanced samples for current epoch "
              f"(class 0: {len([s for s in self.current_samples if self._get_label_for_sample(s) == 0])}, "
              f"class 1: {len([s for s in self.current_samples if self._get_label_for_sample(s) == 1])})")
    
    def _get_label_for_sample(self, sample: Tuple[int, int, int]) -> int:
        """Get label for a sample tuple (img_idx, y, x)."""
        img_idx, y, x = sample
        label = self.gpu_dataset.labels[img_idx, :, y, x]
        if label.ndim > 0:
            label = label[0]
        return int(label.item() > 0.5)
    
    def new_epoch(self):
        """Call this at the start of each epoch to generate new balanced samples."""
        self._generate_epoch_samples()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a patch and its label."""
        img_idx, y_center, x_center = self.current_samples[idx]
        
        # Extract patch
        y_start = y_center - self.half
        y_end = y_center + self.half + 1
        x_start = x_center - self.half
        x_end = x_center + self.half + 1
        
        patch = self.gpu_dataset.images[img_idx, :, y_start:y_end, x_start:x_end]
        
        # Get label
        label = self.gpu_dataset.labels[img_idx, :, y_center, x_center]
        if label.ndim > 0:
            label = label[0]
        label = (label > 0.5).long()
        
        # Apply transform if specified
        if self.transform:
            patch = self.transform(patch)
        
        return patch, label
    
    def __len__(self) -> int:
        """Return number of samples in current epoch."""
        return len(self.current_samples)


class OptimizedTrainer:
    """
    Optimized trainer that uses fast patch sampling and efficient training loops.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the trainer with configuration."""
        self.config = Config.load(config_path)
        self.device = self.config["device"] if torch.cuda.is_available() and self.config["device"] == "cuda" else "cpu"
        
        print(f"Initializing OptimizedTrainer on device: {self.device}")
        
        # Training parameters
        self.patch_size = self.config["n"]
        self.batch_size = self.config["batch_size"]
        self.num_epochs = self.config["num_epochs"]
        self.early_stopping_patience = self.config["early_stopping_patience"]
        self.total_patches = self.config["total_patches"]
        self.seed = self.config["seed"]
        self.resize_shape = tuple(self.config["resize_shape"])
        self.class_weights = torch.tensor(self.config["class_weights"], device=self.device)
        
        # Data splits
        train_split, val_split, test_split = self.config["train_val_test_split"]
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Model paths
        self.model_load_path = self.config["model_load_path"]
        
        # Initialize components
        self.model: Optional[DLModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.gpu_dataset: Optional[GPUMappedDataset] = None
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []
    
    @contextmanager
    def timer(self, description: str):
        """Context manager for timing operations."""
        start_time = time.time()
        yield
        end_time = time.time()
        print(f"{description}: {end_time - start_time:.2f}s")
    
    def setup_model(self):
        """Initialize model, loss function, and optimizer."""
        with self.timer("Model setup"):
            self.model = DLModel()
            self.model.to(self.device)
            
            # Load pretrained weights if available
            if os.path.exists(self.model_load_path):
                try:
                    self.model.load_state_dict(torch.load(self.model_load_path, map_location=self.device))
                    print(f"Loaded pretrained weights from {self.model_load_path}")
                except Exception as e:
                    print(f"Failed to load weights: {e}. Starting from scratch.")
            
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
            
            print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def setup_data(self):
        """Setup GPU dataset and split image indices."""
        with self.timer("Data setup"):
            # Get dataset
            dataset_tuples = DatasetSupplier.get_dataset()
            print(f"Found {len(dataset_tuples)} complete image sets")
            
            # Load to GPU
            self.gpu_dataset = GPUMappedDataset(
                dataset_tuples,
                device=self.device,
                scale_shape=self.resize_shape,
                picture_transform=gcn_pil_to_tensor_transform
            )
            print("All data loaded to GPU")
            
            # Split image indices
            num_images = len(dataset_tuples)
            image_indices = np.arange(num_images)
            np.random.seed(self.seed)
            np.random.shuffle(image_indices)
            
            n_train = int(self.train_split * num_images)
            n_val = max(int(self.val_split * num_images), 1)
            n_test = num_images - n_train - n_val
            
            self.train_indices = image_indices[:n_train].tolist()
            self.val_indices = image_indices[n_train:n_train + n_val].tolist()
            self.test_indices = image_indices[n_train + n_val:].tolist()
            
            print(f"Data split - Train: {len(self.train_indices)}, "
                  f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
            
            # Print which images are in each split
            print("Train images:", [os.path.basename(dataset_tuples[i][0]) for i in self.train_indices])
            print("Val images:", [os.path.basename(dataset_tuples[i][0]) for i in self.val_indices])
            print("Test images:", [os.path.basename(dataset_tuples[i][0]) for i in self.test_indices])
    
    def fast_tensor_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fast tensor-based data augmentation."""
        # Random horizontal flip
        if torch.rand(1, device=x.device) < 0.5:
            x = torch.flip(x, dims=[2])
        
        # Random vertical flip
        if torch.rand(1, device=x.device) < 0.5:
            x = torch.flip(x, dims=[1])
        
        # Random rotation (90 degree increments)
        if torch.rand(1, device=x.device) < 0.5:
            k = int(torch.randint(1, 4, (1,), device=x.device).item())
            x = torch.rot90(x, k, dims=[1, 2])
        
        return x
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, FastPatchSampler, FastPatchSampler, FastPatchSampler]:
        """Create optimized data loaders using FastPatchSampler."""
        with self.timer("DataLoader creation"):
            # Ensure all necessary components are initialized
            assert self.gpu_dataset is not None, "GPU dataset must be initialized before creating data loaders"
            assert len(self.train_indices) > 0, "Train indices must be initialized"
            assert len(self.val_indices) > 0, "Val indices must be initialized"
            assert len(self.test_indices) > 0, "Test indices must be initialized"
            
            # Training dataset with augmentation
            train_dataset = FastPatchSampler(
                self.gpu_dataset,
                self.train_indices,
                patch_size=self.patch_size,
                samples_per_epoch=self.total_patches,
                transform=self.fast_tensor_transform,
                device=self.device
            )
            
            # Validation dataset (smaller, no augmentation)
            val_dataset = FastPatchSampler(
                self.gpu_dataset,
                self.val_indices,
                patch_size=self.patch_size,
                samples_per_epoch=self.total_patches // 8,
                transform=None,
                device=self.device
            )
            
            # Test dataset (smaller, no augmentation)
            test_dataset = FastPatchSampler(
                self.gpu_dataset,
                self.test_indices,
                patch_size=self.patch_size,
                samples_per_epoch=self.total_patches // 8,
                transform=None,
                device=self.device
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=0,  # Keep 0 for GPU tensors
                pin_memory=False,  # Data already on GPU
                drop_last=True  # For consistent batch sizes
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=0, 
                pin_memory=False
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=False
            )
            
            print(f"Created data loaders - Train: {len(train_dataset)}, "
                  f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Data is already on GPU from GPUMappedDataset
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on data loader."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                # Data is already on GPU
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self):
        """Main training loop with early stopping."""
        print("Starting optimized training...")
        
        # Setup everything
        self.setup_model()
        self.setup_data()
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = self.create_data_loaders()
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None
        
        # Training metrics
        train_metrics = {'loss': 0.0, 'accuracy': 0.0}
        val_metrics = {'loss': 0.0, 'accuracy': 0.0}
        
        total_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Generate new epoch samples
            train_dataset.new_epoch()
            val_dataset.new_epoch()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s)")
            print(f"Train - Loss: {train_metrics['loss']:.6f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val   - Loss: {val_metrics['loss']:.6f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                if self.model is not None:
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    break
            
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Restore best weights
        if best_state_dict is not None and self.model is not None:
            self.model.load_state_dict(best_state_dict)
            print("Restored best model weights")
        
        # Save model
        if self.model is not None:
            torch.save(self.model.state_dict(), self.model_load_path)
            print(f"Model saved to {self.model_load_path}")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        # Final evaluation
        print("\nFinal Evaluation")
        print("=" * 50)
        
        final_train_metrics = self.evaluate(train_loader)
        final_val_metrics = self.evaluate(val_loader)
        final_test_metrics = self.evaluate(test_loader)
        
        print(f"Final Train - Loss: {final_train_metrics['loss']:.6f}, Acc: {final_train_metrics['accuracy']:.2f}%")
        print(f"Final Val   - Loss: {final_val_metrics['loss']:.6f}, Acc: {final_val_metrics['accuracy']:.2f}%")
        print(f"Final Test  - Loss: {final_test_metrics['loss']:.6f}, Acc: {final_test_metrics['accuracy']:.2f}%")
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': final_test_metrics,
            'best_val_loss': best_val_loss,
            'total_time': total_time
        }


def main():
    """Main entry point for optimized training."""
    print("=" * 60)
    print("Optimized Retinal Vessel Segmentation Training")
    print("=" * 60)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer and run training
    trainer = OptimizedTrainer()
    results = trainer.train()
    
    print("\nTraining completed successfully!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Total training time: {results['total_time']:.1f}s")


if __name__ == "__main__":
    main()

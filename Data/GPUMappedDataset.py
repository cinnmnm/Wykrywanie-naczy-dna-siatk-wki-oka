import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from Data.Preprocessing import ImagePreprocessing
import torchvision.transforms as T

class GPUMappedDataset(Dataset):
    def __init__(self, dataset_tuples, device='cuda', scale_shape=(256, 256), picture_transform=None):
        """
        Loads and preprocesses all images/masks/labels to GPU memory for efficient patch extraction.
        
        Args:
            dataset_tuples: list of (basename, image_path, label_path, mask_path)
            device: GPU device to load data onto
            scale_shape: target shape for resizing
            picture_transform: optional transform function (applied on GPU tensors)
        """
        print(f"[GPUMappedDataset] Initializing with {len(dataset_tuples)} images, device={device}")
        self.device = device
        self.scale_shape = scale_shape
        self.picture_transform = picture_transform
        
        # Extract paths
        self.basenames = [t[0] for t in dataset_tuples]
        image_paths = [t[1] for t in dataset_tuples]
        label_paths = [t[2] for t in dataset_tuples]
        mask_paths = [t[3] for t in dataset_tuples]
        
        # Preprocess on CPU first, then load to GPU
        print("[GPUMappedDataset] Preprocessing images on CPU...")
        images_np = [ImagePreprocessing.preprocess_image(p, scale_shape, color_adjustment=True, normalization=True) for p in image_paths]
        print("[GPUMappedDataset] Preprocessing masks on CPU...")
        masks_np = [ImagePreprocessing.preprocess_mask_or_label(p, scale_shape) for p in mask_paths]
        print("[GPUMappedDataset] Preprocessing labels on CPU...")
        labels_np = [ImagePreprocessing.preprocess_mask_or_label(p, scale_shape) for p in label_paths]
          # Convert to tensors and load to GPU
        print("[GPUMappedDataset] Loading to GPU memory...")
        # Images: [N, C, H, W] format
        images_tensors = []
        for img_np in images_np:
            if img_np.shape[-1] == 3:  # [H, W, C] -> [C, H, W]
                img_np = img_np.transpose(2, 0, 1)
            tensor = torch.from_numpy(img_np).float().to(device)
            # Note: picture_transform is not applied here since preprocessing is already done
            # If you need additional GPU-based transforms, apply them separately
            images_tensors.append(tensor)
        self.images = torch.stack(images_tensors)
          # Masks and labels: [N, 1, H, W] format (add channel dimension for compatibility)
        self.masks = torch.stack([torch.from_numpy(mask).unsqueeze(0).long().to(device) for mask in masks_np])
        self.labels = torch.stack([torch.from_numpy(label).unsqueeze(0).long().to(device) for label in labels_np])
        
        print(f"[GPUMappedDataset] Loaded shapes - Images: {self.images.shape}, Masks: {self.masks.shape}, Labels: {self.labels.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.masks[idx]
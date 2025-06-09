import os
import numpy as np
import torch
from torch.utils.data import Dataset
from Util.ImageLoader import ImageLoader
from PIL import Image
from Data.Preprocessing import ImagePreprocessing

class PatchIndexFinder:
    """
    Identifies all valid patch indices (img_idx, center_y, center_x, label) for a given patch size and mask.
    Saves the indices and labels to a file for later use.
    Optionally applies an image_transform to images, labels, and masks before index finding.
    """
    def __init__(self, image_dir, n=27, output_file="patch_indices_labels.npy", image_transform=None):
        self.image_dir = image_dir
        self.n = n
        self.output_file = output_file
        self.image_transform = image_transform

    def find_and_save(self):
        image_paths = [os.path.join(self.image_dir, "pictures", fname) for fname in os.listdir(os.path.join(self.image_dir, "pictures"))]
        labels_paths = [
            os.path.join(self.image_dir, "manual", os.path.basename(p).replace(".JPG", ".tif").replace(".jpg", ".tif"))
            for p in image_paths
        ]
        masks_paths = [
            os.path.join(self.image_dir, "mask", os.path.basename(p).replace(".JPG", "_mask.tif").replace(".jpg", "_mask.tif"))
            for p in image_paths
        ]
        images_list = ImageLoader.load_images(image_paths)
        labels_list = ImageLoader.load_images(labels_paths)
        masks_list = ImageLoader.load_images(masks_paths)
        # Apply image_transform if provided
        if self.image_transform is not None:
            images_list = [self.image_transform(img) for img in images_list]
            labels_list = [self.image_transform(lbl) for lbl in labels_list]
            masks_list = [self.image_transform(msk) for msk in masks_list]
        images = np.stack(images_list)
        labels = np.stack(labels_list)
        masks = np.stack(masks_list)
        n = self.n
        patch_indices = []  # (img_idx, center_y, center_x, label)
        for i in range(images.shape[0]):
            lbl = labels[i]
            msk = masks[i]
            h, w = lbl.shape[:2]
            for center_y in range(n // 2, h - n // 2):
                for center_x in range(n // 2, w - n // 2):
                    patch_x_start = center_x - n // 2
                    patch_x_end = center_x + n // 2 + 1
                    patch_y_start = center_y - n // 2
                    patch_y_end = center_y + n // 2 + 1
                    mask_patch = msk[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
                    if np.all(mask_patch):
                        patch_label_val = lbl[center_y, center_x]
                        if hasattr(patch_label_val, 'max') and patch_label_val.max() > 1:
                            patch_label_val = patch_label_val / 255.0
                        patch_label_binary = int(patch_label_val > 0.5)
                        patch_indices.append((i, center_y, center_x, patch_label_binary))
            print(f"Image {i}: found {sum(1 for idx in patch_indices if idx[0]==i)} valid patches.")
        patch_indices_np = np.array(patch_indices, dtype=np.int32)
        np.save(self.output_file, patch_indices_np)
        print(f"Saved all valid patch indices and labels to {self.output_file}. Total patches: {len(patch_indices_np)}.")

class PatchOnDemandDataset(Dataset):
    """
    PyTorch Dataset that holds all images in memory and fetches patches on demand using precomputed indices.
    Optionally balances the dataset and supports transforms.
    Allows for image-level transforms at load time (e.g., normalization, resizing).
    Pass image_transform to resize/scale images before patch extraction.
    """
    def __init__(self, patch_indices_file, image_dir, n=27, balance=False, target_num=None, transform=None, seed=42, indices=None, image_transform=None):
        self.patch_info = np.load(patch_indices_file)
        self.image_dir = image_dir
        self.n = n
        self.transform = transform
        # Subsample indices for split
        if indices is not None:
            self.patch_info = self.patch_info[indices]
        # Balance if requested (undersample majority class)
        if balance:
            np.random.seed(seed)
            class0 = np.where(self.patch_info[:, 3] == 0)[0]
            class1 = np.where(self.patch_info[:, 3] == 1)[0]
            min_len = min(len(class0), len(class1))
            if target_num is not None:
                min_len = min(min_len, target_num // 2)
            np.random.shuffle(class0)
            np.random.shuffle(class1)
            balanced_indices = np.concatenate([class0[:min_len], class1[:min_len]])
            np.random.shuffle(balanced_indices)
            self.patch_info = self.patch_info[balanced_indices]
        elif target_num is not None and len(self.patch_info) > target_num:
            np.random.seed(seed)
            chosen = np.random.choice(len(self.patch_info), target_num, replace=False)
            self.patch_info = self.patch_info[chosen]
        # Load all images into memory for fast patch extraction
        image_paths = [os.path.join(image_dir, "pictures", fname) for fname in os.listdir(os.path.join(image_dir, "pictures"))]
        images_list = ImageLoader.load_images(image_paths)
        if image_transform is not None:
            self.images = np.stack([image_transform(img) for img in images_list])
        else:
            self.images = np.stack(images_list)
    def __len__(self):
        return len(self.patch_info)
    def __getitem__(self, idx):
        img_idx, center_y, center_x, label = self.patch_info[idx]
        n = self.n
        img = self.images[img_idx]
        patch = img[center_y - n // 2:center_y + n // 2 + 1, center_x - n // 2:center_x + n // 2 + 1, :]
        if self.transform is not None:
            if isinstance(patch, np.ndarray):
                patch = Image.fromarray(patch)
            patch = self.transform(patch)
            patch_tensor = patch.float() if torch.is_tensor(patch) else torch.from_numpy(patch).float()
        else:
            patch = np.transpose(patch, (2, 0, 1))
            patch_tensor = torch.from_numpy(patch).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return patch_tensor, label_tensor
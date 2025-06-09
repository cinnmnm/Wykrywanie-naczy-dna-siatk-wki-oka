"""
Deep Learning Model for Retinal Vessel Segmentation

This module implements a CNN-based approach for patch-level vessel segmentation.
Key features:
- Automatic patch discovery with caching support
- GPU-accelerated data pipeline
- Balanced class sampling
- Early stopping with validation monitoring

Patch Index Caching:
The system can load pre-computed patch indices from config['patch_indices_file'] 
to skip the time-consuming patch discovery process. If the file doesn't exist or 
can't be loaded, patch indices are computed from scratch and saved for future use.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as T
import os

from Util.config import Config
from Util.Evaluate import Evaluate
from Data.Preprocessing import ImagePreprocessing
from Data.GPUMappedDataset import GPUMappedDataset
from Data.GPUPatchSampler import GPUPatchSampler
from Data.DatasetSupplier import DatasetSupplier

class DLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_size = 64
        self.conv2_size = 128
        self.fc_size = 512
        self.input_channels = 3

        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 128, self.fc_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.fc_size, 2)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

config = Config.load("config.yaml")

n = config["n"]
device = config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
model_load_path = config["model_load_path"]
patch_indices_file = config["patch_indices_file"]
image_dir = config["image_dir"]
total_patches = config["total_patches"]
seed = config["seed"]
resize_shape = tuple(config["resize_shape"])
class_weights = torch.tensor(config["class_weights"], device=device)
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
early_stopping_patience = config["early_stopping_patience"]
train_split, val_split, test_split = config["train_val_test_split"]

def train_model(model: nn.Module, train_loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, num_epochs: int=10, device: str='cpu', val_loader=None, early_stopping_patience=5):
    model.train()
    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
        epoch_duration = time.time() - epoch_start_time

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0
        val_loss, val_acc = None, None
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, device=device, silent=True)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Duration: {epoch_duration:.2f}s")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                    break
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("Best model parameters restored (early stopping).")

def evaluate_model(model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module, device: str='cpu', silent=False):
    model.eval()
    total_loss = 0
    #correct = 0
    #total = 0
    all_targets = []  # To store targets for evaluation
    all_preds = []

    with torch.no_grad():
        for inputs, targets in test_loader:  # Assuming masks are provided in the DataLoader
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            #correct += (predicted == targets).sum().item()
            #total += targets.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    all_targets_np = np.concatenate(all_targets)
    all_preds_np = np.concatenate(all_preds)

    accuracy    = Evaluate.accuracy(all_targets_np, all_preds_np)
    sensitivity = Evaluate.sensitivity(all_targets_np, all_preds_np)
    specificity = Evaluate.specificity(all_targets_np, all_preds_np)

    if not all_targets:
        if not silent:
            print(f"Test Loss: {avg_loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print("Warning: No targets found in test_loader. Skipping precision, recall, F1 calculation.")
            print("Sensitivity: N/A, Specificity: N/A, Avg F1: N/A (no targets)")
        return avg_loss, accuracy

    if not silent:
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

    return avg_loss, accuracy

def get_patches_for_images(img_indices, all_patch_info, total_patches, seed, force_1_to_1_balance=True):
    # Select only patches from the given image indices
    mask = np.isin(all_patch_info[:, 0], img_indices)
    patch_info = all_patch_info[mask]
    # Balance classes
    class0 = patch_info[patch_info[:, 3] == 0]
    class1 = patch_info[patch_info[:, 3] == 1]
    limit_per_class = total_patches // 2 if total_patches > 0 else float('inf')
    num_to_sample_c0 = min(len(class0), limit_per_class)
    num_to_sample_c1 = min(len(class1), limit_per_class)
    if force_1_to_1_balance:
        final_count_each_class = min(num_to_sample_c0, num_to_sample_c1)
        num_to_sample_c0 = final_count_each_class
        num_to_sample_c1 = final_count_each_class
    np.random.seed(seed)
    np.random.shuffle(class0)
    np.random.shuffle(class1)
    balanced_class0_data = class0[:num_to_sample_c0]
    balanced_class1_data = class1[:num_to_sample_c1]
    all_balanced_patches_data = np.concatenate([balanced_class0_data, balanced_class1_data])
    np.random.shuffle(all_balanced_patches_data)
    return all_balanced_patches_data

def get_patches_for_images_o(img_indices, all_patch_info, total_patches, seed, force_1_to_1_balance=True):
    rng = np.random.default_rng(seed)

    if all_patch_info.shape[0] == 0 or len(img_indices) == 0:
        return np.empty((0, all_patch_info.shape[1]), dtype=all_patch_info.dtype)

    # Optimized patch selection:
    # Assumes all_patch_info[:, 0] is sorted by image index.
    # This is generally true if GPUPatchSampler iterates through images sequentially.
    
    # Find unique image indices in all_patch_info and their start/end positions
    # This is efficient if all_patch_info[:, 0] is sorted.
    unique_img_ids_in_all, first_occurrence_indices = np.unique(all_patch_info[:, 0], return_index=True)
    
    # Determine end positions for each block of image_ids
    # The end position of a block for image_id[i] is the start position of image_id[i+1]
    # or the total length of all_patch_info for the last unique image_id.
    end_occurrence_indices = np.concatenate((first_occurrence_indices[1:], [all_patch_info.shape[0]]))
    
    # Create a lookup map from image_id to its slice [start, end) in all_patch_info
    image_id_to_slice_map = {
        uid: slice(start, end) for uid, start, end in zip(unique_img_ids_in_all, first_occurrence_indices, end_occurrence_indices)
    }

    # Collect patches for the requested img_indices
    patches_list_for_selection = []
    for img_idx in np.unique(img_indices): # Use unique img_indices, sort not strictly needed for map lookup
        if img_idx in image_id_to_slice_map:
            patches_list_for_selection.append(all_patch_info[image_id_to_slice_map[img_idx]])
    
    if not patches_list_for_selection:
        return np.empty((0, all_patch_info.shape[1]), dtype=all_patch_info.dtype)
    
    patch_info_for_selected_images = np.concatenate(patches_list_for_selection)

    if patch_info_for_selected_images.shape[0] == 0:
        return np.empty((0, all_patch_info.shape[1]), dtype=all_patch_info.dtype)

    # Separate classes from the selected patches
    class0_patches = patch_info_for_selected_images[patch_info_for_selected_images[:, 3] == 0]
    class1_patches = patch_info_for_selected_images[patch_info_for_selected_images[:, 3] == 1]

    # Determine number of samples per class
    if total_patches <= 0: # If non-positive total_patches, take as many as possible (balanced)
        limit_per_class = float('inf')
    else:
        limit_per_class = total_patches // 2

    num_to_sample_c0 = min(len(class0_patches), limit_per_class if limit_per_class != float('inf') else len(class0_patches))
    num_to_sample_c1 = min(len(class1_patches), limit_per_class if limit_per_class != float('inf') else len(class1_patches))


    if force_1_to_1_balance:
        final_count_each_class = min(num_to_sample_c0, num_to_sample_c1)
        num_to_sample_c0 = final_count_each_class
        num_to_sample_c1 = final_count_each_class
    
    # Sample patches using np.random.choice (more efficient than full shuffle)
    sampled_class0_data = np.empty((0, all_patch_info.shape[1]), dtype=all_patch_info.dtype)
    if len(class0_patches) > 0 and num_to_sample_c0 > 0:
        indices_c0 = rng.choice(len(class0_patches), size=int(num_to_sample_c0), replace=False)
        sampled_class0_data = class0_patches[indices_c0]

    sampled_class1_data = np.empty((0, all_patch_info.shape[1]), dtype=all_patch_info.dtype)
    if len(class1_patches) > 0 and num_to_sample_c1 > 0:
        indices_c1 = rng.choice(len(class1_patches), size=int(num_to_sample_c1), replace=False)
        sampled_class1_data = class1_patches[indices_c1]
    
    # Combine and shuffle
    if sampled_class0_data.shape[0] == 0 and sampled_class1_data.shape[0] == 0:
        return np.empty((0, all_patch_info.shape[1]), dtype=all_patch_info.dtype)
    
    all_balanced_patches_data = np.concatenate((sampled_class0_data, sampled_class1_data))
    
    rng.shuffle(all_balanced_patches_data) # Shuffle the final combined list

    return all_balanced_patches_data



def gcn_pil_to_tensor_transform(pil_img):
        if pil_img.mode != 'RGB':
            pil_img_rgb = pil_img.convert('RGB')
        else:
            pil_img_rgb = pil_img
        np_img = np.array(pil_img_rgb)
        gcn_img = ImagePreprocessing.global_contrast_normalization(np_img)
        gcn_img_uint8 = np.clip((gcn_img - gcn_img.min()) / (gcn_img.max() - gcn_img.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
        return T.ToTensor()(gcn_img_uint8)

# --- Fast tensor-based patch transform (no PIL, all on GPU) ---
def fast_patch_transform(x):
    # x: [C, H, W] tensor, already on GPU
    if torch.rand(1, device=x.device) < 0.5:
        x = torch.flip(x, dims=[2])  # Horizontal flip
    if torch.rand(1, device=x.device) < 0.5:
        x = torch.flip(x, dims=[1])  # Vertical flip
    return x


if __name__ == "__main__":
    overall_start_time = time.time()
    last_time = overall_start_time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    continue_training = True
    current_time = time.time()
    print(f"Time for device setup: {current_time - last_time:.2f}s")
    last_time = current_time

    model = DLModel()
    model.to(device)
    current_time = time.time()
    print(f"Time for model initialization and to(device): {current_time - last_time:.2f}s")
    last_time = current_time

    if continue_training:
        if os.path.exists(model_load_path):
            print(f"Continue training enabled. Attempting to load weights from: {model_load_path}")
            try:
                model.load_state_dict(torch.load(model_load_path, map_location=device))
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading weights from {model_load_path}: {e}. Training will start from scratch.")
        else:
            print(f"Continue training enabled, but weights file not found: {model_load_path}. Training will start from scratch.")
    else:
        print("Starting training from scratch. Not loading any pre-trained weights.")
    current_time = time.time()
    print(f"Time for model loading (if any): {current_time - last_time:.2f}s")
    last_time = current_time

    print(model)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=0.0005)
    current_time = time.time()
    print(f"Time for loss_fn and optimizer setup: {current_time - last_time:.2f}s")
    last_time = current_time

    # Patch-level transforms (use fast tensor-based transform for training)
    transform = fast_patch_transform
    val_transform = None  # No transform for validation/test
    current_time = time.time()
    print(f"Time for transform setup: {current_time - last_time:.2f}s")
    last_time = current_time

    # --- Dynamic Data Pipeline ---
    print("Starting dynamic data pipeline...")
    dataset_tuples = DatasetSupplier.get_dataset()
    print(f"Dynamically found {len(dataset_tuples)} complete sets of image, mask, and label files.")
    current_time = time.time()
    print(f"Time for DatasetSupplier.get_dataset(): {current_time - last_time:.2f}s")
    last_time = current_time

    print("Loading and preprocessing data to GPU using GPUMappedDataset...")
    gpu_dataset = GPUMappedDataset(
        dataset_tuples,
        device=device,
        scale_shape=resize_shape,
        picture_transform=gcn_pil_to_tensor_transform
    )
    print("All source images, masks, and labels loaded to GPU.")
    current_time = time.time()
    print(f"Time for GPUMappedDataset creation: {current_time - last_time:.2f}s")
    last_time = current_time

    # --- New: Split images first, then collect patches for each split ---
    print("Splitting images...")
    num_images = len(dataset_tuples)
    image_indices = np.arange(num_images)
    np.random.seed(seed)
    np.random.shuffle(image_indices)
    n_train_imgs = int(train_split * num_images)
    n_val_imgs = max(int(val_split * num_images), 1)
    n_test_imgs = num_images - n_train_imgs - n_val_imgs
    print (f"Total images: {num_images}, Train: {n_train_imgs}, Val: {n_val_imgs}, Test: {n_test_imgs}")
    train_img_indices = image_indices[:n_train_imgs]
    val_img_indices = image_indices[n_train_imgs:n_train_imgs + n_val_imgs]
    test_img_indices = image_indices[n_train_imgs + n_val_imgs:]
    current_time = time.time()
    print(f"Time for image index splitting: {current_time - last_time:.2f}s")
    last_time = current_time

    print("Train image files:")
    for idx in train_img_indices:
        print(f"  {os.path.basename(dataset_tuples[idx][0])}")

    print("Validation image files:")
    for idx in val_img_indices:
        print(f"  {os.path.basename(dataset_tuples[idx][0])}")

    print("Test image files:")
    for idx in test_img_indices:
        print(f"  {os.path.basename(dataset_tuples[idx][0])}")    # Load patch indices from file if available, otherwise compute them
    patch_info_from_gpu = None
    if patch_indices_file and os.path.exists(patch_indices_file):
        print(f"Loading pre-computed patch indices from: {patch_indices_file}")
        try:
            patch_info_from_gpu = np.load(patch_indices_file)
            print(f"Successfully loaded {len(patch_info_from_gpu)} patch indices from file.")
            
            # Validate loaded patch indices format
            if patch_info_from_gpu.size == 0:
                raise ValueError("Loaded patch indices file is empty.")
            if patch_info_from_gpu.ndim == 1 and patch_info_from_gpu.shape[0] == 4:
                patch_info_from_gpu = patch_info_from_gpu.reshape(1, 4)
            elif patch_info_from_gpu.ndim != 2 or patch_info_from_gpu.shape[1] != 4:
                raise ValueError(f"Invalid patch indices format. Expected shape (N, 4), got {patch_info_from_gpu.shape}")
                
            current_time = time.time()
            print(f"Time for loading patch indices from file: {current_time - last_time:.2f}s")
            last_time = current_time
            
        except Exception as e:
            print(f"Error loading patch indices from {patch_indices_file}: {e}")
            print("Falling back to computing patch indices...")
            patch_info_from_gpu = None
    else:
        if patch_indices_file:
            print(f"Patch indices file not found: {patch_indices_file}")
        else:
            print("No patch indices file specified in config.")
        print("Computing patch indices from scratch...")
    
    # Compute patch indices if not loaded from file
    if patch_info_from_gpu is None:
        print("Discovering all valid patches using GPUPatchSampler from GPU data...")
        helper_sampler = GPUPatchSampler(gpu_dataset, patch_size=n, mask_threshold=0.5, balance=False)
        patch_info_from_gpu = np.array(helper_sampler.valid_indices)
        
        if patch_info_from_gpu.size == 0:
            raise ValueError("No valid patches found by GPUPatchSampler. Check mask_threshold, data, or mask values.")
        if patch_info_from_gpu.ndim == 1 and patch_info_from_gpu.shape[0] == 4:
            patch_info_from_gpu = patch_info_from_gpu.reshape(1, 4)
        elif patch_info_from_gpu.ndim != 2 or patch_info_from_gpu.shape[1] != 4:
            raise ValueError(f"Unexpected shape for patch_info_from_gpu: {patch_info_from_gpu.shape}")
          # Save computed patch indices for future use
        if patch_indices_file:
            try:
                # Create directory if it doesn't exist (handle case where no directory is specified)
                patch_dir = os.path.dirname(patch_indices_file)
                if patch_dir:  # Only create directory if there's actually a directory path
                    os.makedirs(patch_dir, exist_ok=True)
                np.save(patch_indices_file, patch_info_from_gpu)
                print(f"Saved computed patch indices to: {patch_indices_file}")
            except Exception as e:
                print(f"Warning: Could not save patch indices to {patch_indices_file}: {e}")
                print("Patch discovery will need to be repeated on next run.")
        current_time = time.time()
        print(f"Time for discovering all patches: {current_time - last_time:.2f}s")
        last_time = current_time

    # Summary of patch discovery process
    print(f"\n--- Patch Discovery Summary ---")
    print(f"Total valid patches found: {len(patch_info_from_gpu):,}")
    print(f"Patch format: {patch_info_from_gpu.shape} (img_idx, y, x, label)")
    unique_images = len(np.unique(patch_info_from_gpu[:, 0]))
    print(f"Patches span across {unique_images} images")
    class_0_count = np.sum(patch_info_from_gpu[:, 3] == 0)
    class_1_count = np.sum(patch_info_from_gpu[:, 3] == 1)
    print(f"Class distribution: Background={class_0_count:,}, Vessel={class_1_count:,}")
    print(f"Class balance ratio: {class_1_count/class_0_count:.3f}" if class_0_count > 0 else "Class balance ratio: N/A")
    print(f"--------------------------------\n")

    # Split patches by image split
    print("Splitting patches for train, validation, and test sets...")
    train_patches = get_patches_for_images(train_img_indices, patch_info_from_gpu, total_patches, seed, force_1_to_1_balance=config.get("force_1_to_1_gpu_balance", True))
    val_patches = get_patches_for_images(val_img_indices, patch_info_from_gpu, total_patches // 8, seed+1, force_1_to_1_balance=config.get("force_1_to_1_gpu_balance", True))
    test_patches = get_patches_for_images(test_img_indices, patch_info_from_gpu, total_patches // 8, seed+2, force_1_to_1_balance=config.get("force_1_to_1_gpu_balance", True))
    current_time = time.time()
    print(f"Time for splitting patches: {current_time - last_time:.2f}s")
    last_time = current_time

    print(f"Train images: {len(train_img_indices)}, val images: {len(val_img_indices)}, test images: {len(test_img_indices)}")
    print(f"Train patches: {len(train_patches)}, val patches: {len(val_patches)}, test patches: {len(test_patches)}")

    train_coords_for_sampler = [tuple(row[:3]) for row in train_patches]
    val_coords_for_sampler = [tuple(row[:3]) for row in val_patches]
    test_coords_for_sampler = [tuple(row[:3]) for row in test_patches]

    print("Creating GPUPatchSampler datasets...")
    train_dataset = GPUPatchSampler(gpu_dataset, patch_size=n,
                                    precomputed_indices=train_coords_for_sampler,
                                    transform=transform,
                                    balance=False)
    val_dataset = GPUPatchSampler(gpu_dataset, patch_size=n,
                                  precomputed_indices=val_coords_for_sampler,
                                  transform=val_transform,
                                  balance=False)
    test_dataset = GPUPatchSampler(gpu_dataset, patch_size=n,
                                   precomputed_indices=test_coords_for_sampler,
                                   transform=val_transform,
                                   balance=False)
    current_time = time.time()
    print(f"Time for creating GPUPatchSampler datasets: {current_time - last_time:.2f}s")
    last_time = current_time

    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    current_time = time.time()
    print(f"Time for creating DataLoaders: {current_time - last_time:.2f}s")
    last_time = current_time

    print(f"Created DataLoaders: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    if n < 16:
        print(f"Warning: Patch size {n}x{n} might be too small for the current model architecture, potentially leading to dimension errors.")

    print("Starting model training...")
    train_model(model, train_loader, loss_fn, optimizer, num_epochs=num_epochs, device=device, val_loader=val_loader, early_stopping_patience=early_stopping_patience)
    current_time = time.time()
    print(f"Time for model training: {current_time - last_time:.2f}s")
    last_time = current_time

    torch.save(model.state_dict(), model_load_path)
    print(f"Model parameters saved to {model_load_path}")
    current_time = time.time()
    print(f"Time for saving model: {current_time - last_time:.2f}s")
    last_time = current_time

    print("Starting model evaluation...")
    evaluate_model(model, test_loader, loss_fn, device=device)
    current_time = time.time()
    print(f"Time for model evaluation: {current_time - last_time:.2f}s")
    last_time = current_time

    overall_end_time = time.time()
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f}s")
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars

# Define constants
IMG_WIDTH = 256  # Resize to this width
IMG_HEIGHT = 256 # Resize to this height
IMG_CHANNELS = 3 # Input images are RGB

# Paths (Update these if your folder structure is different or on a different drive)
BASE_PATH = Path('images')
PICTURES_PATH = BASE_PATH / 'pictures'
MASK_PATH = BASE_PATH / 'mask'
MANUAL_PATH = BASE_PATH / 'manual'

# Helper function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    img = Image.open(image_path)
    if img.mode != 'RGB': # Ensure image is RGB
        img = img.convert('RGB')
    img = img.resize(target_size) # Default resampling is usually Bicubic or Bilinear
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array.astype(np.float32)

# Helper function to load and preprocess masks/labels
def load_and_preprocess_mask(mask_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    mask = Image.open(mask_path)
    if mask.mode == 'RGB': # Some masks might be RGB (0,0,0) or (255,255,255)
        mask = mask.convert('L') # Convert to grayscale
    mask = mask.resize(target_size, Image.NEAREST) # Use NEAREST to avoid new pixel values
    mask_array = np.array(mask)
    # Convert to binary: 1 for vessel/valid ROI, 0 for background/invalid ROI
    binary_mask = (mask_array > 128).astype(np.float32)
    return np.expand_dims(binary_mask, axis=-1) # Add channel dimension (H, W, 1)

def get_file_paths():
    image_files = []
    roi_mask_files = []
    label_files = []

    # Iterate through picture files to ensure matching sets
    for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG'): # Add other extensions if present
        for p_file_path in PICTURES_PATH.glob(ext):
            base_name = p_file_path.stem # e.g., "01_dr"

            # Construct corresponding mask and label file names
            mask_file_name = f"{base_name}_mask.tif"
            roi_mask_file_path = MASK_PATH / mask_file_name

            label_file_name = f"{base_name}.tif" # Assuming manual labels follow this pattern
            label_file_path = MANUAL_PATH / label_file_name

            if roi_mask_file_path.exists() and label_file_path.exists():
                image_files.append(p_file_path)
                roi_mask_files.append(roi_mask_file_path)
                label_files.append(label_file_path)
            else:
                print(f"Warning: Missing ROI mask or label for {p_file_path.name}")
                if not roi_mask_file_path.exists():
                    print(f"  Missing ROI mask: {roi_mask_file_path}")
                if not label_file_path.exists():
                    print(f"  Missing label: {label_file_path}")

    return image_files, roi_mask_files, label_files

class RetinaDataset(Dataset):
    def __init__(self, image_paths, label_paths, roi_mask_paths, target_size=(IMG_HEIGHT, IMG_WIDTH)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.roi_mask_paths = roi_mask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        roi_mask_path = self.roi_mask_paths[idx]

        try:
            img = load_and_preprocess_image(img_path, self.target_size) # (H, W, C)
            roi_mask = load_and_preprocess_mask(roi_mask_path, self.target_size) # (H, W, 1)
            label = load_and_preprocess_mask(label_path, self.target_size) # (H, W, 1)

            masked_label = label * roi_mask # Apply ROI mask

            # Debug: Print unique values for label and mask occasionally
            if np.random.rand() < 0.01:
                print(f"Sample {img_path.name}: label unique={np.unique(label)}, roi_mask unique={np.unique(roi_mask)}, masked_label unique={np.unique(masked_label)}")

            # Convert to PyTorch tensors and permute to (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            masked_label_tensor = torch.from_numpy(masked_label).permute(2, 0, 1)

            return img_tensor, masked_label_tensor
        except Exception as e:
            print(f"Error loading sample {img_path.name}: {e}")
            if idx + 1 < len(self.image_paths):
                return self.__getitem__(idx + 1)
            else:
                raise RuntimeError(f"Could not load last sample {img_path.name} due to {e}")


# U-Net Model Architecture (PyTorch)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.c1_conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)
        self.c1_relu1 = nn.ReLU(inplace=True)
        self.c1_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.c1_relu2 = nn.ReLU(inplace=True)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.c2_relu1 = nn.ReLU(inplace=True)
        self.c2_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c2_relu2 = nn.ReLU(inplace=True)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.c3_relu1 = nn.ReLU(inplace=True)
        self.c3_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c3_relu2 = nn.ReLU(inplace=True)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c4_relu1 = nn.ReLU(inplace=True)
        self.c4_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.c4_relu2 = nn.ReLU(inplace=True)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.c5_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.c5_relu1 = nn.ReLU(inplace=True)
        self.c5_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.c5_relu2 = nn.ReLU(inplace=True)

        # Decoder
        self.u6_upconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 128 (from u6) + 128 (from c4)
        self.c6_relu1 = nn.ReLU(inplace=True)
        self.c6_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.c6_relu2 = nn.ReLU(inplace=True)

        self.u7_upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 64 (from u7) + 64 (from c3)
        self.c7_relu1 = nn.ReLU(inplace=True)
        self.c7_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c7_relu2 = nn.ReLU(inplace=True)

        self.u8_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 32 (from u8) + 32 (from c2)
        self.c8_relu1 = nn.ReLU(inplace=True)
        self.c8_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c8_relu2 = nn.ReLU(inplace=True)

        self.u9_upconv = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 16 (from u9) + 16 (from c1)
        self.c9_relu1 = nn.ReLU(inplace=True)
        self.c9_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.c9_relu2 = nn.ReLU(inplace=True)

        self.outputs = nn.Conv2d(16, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.c1_relu1(self.c1_conv1(x))
        c1 = self.c1_relu2(self.c1_conv2(c1))
        p1 = self.p1(c1)

        c2 = self.c2_relu1(self.c2_conv1(p1))
        c2 = self.c2_relu2(self.c2_conv2(c2))
        p2 = self.p2(c2)

        c3 = self.c3_relu1(self.c3_conv1(p2))
        c3 = self.c3_relu2(self.c3_conv2(c3))
        p3 = self.p3(c3)

        c4 = self.c4_relu1(self.c4_conv1(p3))
        c4 = self.c4_relu2(self.c4_conv2(c4))
        p4 = self.p4(c4)

        # Bottleneck
        c5 = self.c5_relu1(self.c5_conv1(p4))
        c5 = self.c5_relu2(self.c5_conv2(c5))

        # Decoder
        u6 = self.u6_upconv(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6_relu1(self.c6_conv1(u6))
        c6 = self.c6_relu2(self.c6_conv2(c6))

        u7 = self.u7_upconv(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7_relu1(self.c7_conv1(u7))
        c7 = self.c7_relu2(self.c7_conv2(c7))

        u8 = self.u8_upconv(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8_relu1(self.c8_conv1(u8))
        c8 = self.c8_relu2(self.c8_conv2(c8))

        u9 = self.u9_upconv(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9_relu1(self.c9_conv1(u9))
        c9 = self.c9_relu2(self.c9_conv2(c9))

        logits = self.outputs(c9)
        return self.sigmoid(logits)

# Dice Coefficient and Dice Loss (PyTorch)
def dice_coefficient_pytorch(y_pred, y_true, smooth=1e-6):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_pred_f * y_true_f).sum()
    dice = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
    return dice

def dice_loss_pytorch(y_pred, y_true):
    return 1 - dice_coefficient_pytorch(y_pred, y_true)

# BCE + Dice Loss
def bce_dice_loss(y_pred, y_true):
    bce = nn.BCELoss()(y_pred, y_true)
    dice = dice_loss_pytorch(y_pred, y_true)
    return bce + dice

# Visualization function (PyTorch)
def visualize_predictions(model, dataloader, device, num_samples=3, threshold=0.5):
    if len(dataloader.dataset) == 0:
        print("No test data to visualize.")
        return

    num_samples = min(num_samples, len(dataloader.dataset))
    if num_samples == 0:
        print("No samples to visualize.")
        return

    model.eval()
    with torch.no_grad():
        # Get a few samples
        data_iter = iter(dataloader)
        for _ in range(num_samples):
            try:
                sample_imgs_batch, sample_labels_batch = next(data_iter)
            except StopIteration:
                print("Not enough samples in dataloader for visualization.")
                break
            
            # Take the first image from the batch for simplicity
            sample_img_tensor = sample_imgs_batch[0]
            sample_label_tensor = sample_labels_batch[0]

            # Predict
            pred_label_prob_tensor = model(sample_img_tensor.unsqueeze(0).to(device)).squeeze(0) # Add batch, predict, remove batch
            
            # Move to CPU and convert to NumPy
            sample_img_np = sample_img_tensor.cpu().permute(1, 2, 0).numpy() # C,H,W -> H,W,C
            sample_label_np = sample_label_tensor.cpu().squeeze().numpy() # 1,H,W -> H,W
            pred_label_prob_np = pred_label_prob_tensor.cpu().squeeze().numpy() # 1,H,W -> H,W
            
            pred_label_np = (pred_label_prob_np > threshold).astype(np.uint8)

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(sample_img_np)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(sample_label_np, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title(f"Predicted Mask (Thresh: {threshold})")
            plt.imshow(pred_label_np, cmap='gray')
            plt.axis('off')

            plt.show()
    model.train() # Set model back to training mode

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading file paths...")
    image_paths, roi_mask_paths, label_paths = get_file_paths()

    if not image_paths:
        print("No image files found. Please check paths and file extensions.")
        return

    print(f"Found {len(image_paths)} image/mask/label sets.")

    # Split data into training and validation sets (paths)
    img_train, img_val, lbl_train, lbl_val, roi_train, roi_val = train_test_split(
        image_paths, label_paths, roi_mask_paths, test_size=0.2, random_state=42
    )

    if not img_train:
        print("Not enough data to create a training set. Need at least 2 samples for a 0.2 test split.")
        return

    print(f"Training samples: {len(img_train)}, Validation samples: {len(img_val)}")

    # Create Datasets and DataLoaders
    train_dataset = RetinaDataset(img_train, lbl_train, roi_train)
    val_dataset = RetinaDataset(img_val, lbl_val, roi_val)

    # Check if datasets are empty after potential loading errors in __getitem__
    if len(train_dataset) == 0 or len(val_dataset) == 0:
         print("One of the datasets is empty after attempting to load samples. Check for errors during data loading.")
         # Further filter out problematic paths if necessary, or ensure __getitem__ handles errors robustly
         # For now, we'll proceed, but DataLoader might complain if dataset is truly empty.
         # A more robust solution would be to pre-filter paths that cause loading errors.
         # Or, ensure RetinaDataset returns None and collate_fn in DataLoader handles it.
         # For simplicity, we assume some valid data can be loaded.
         if len(train_dataset) == 0:
             print("Training dataset is empty. Cannot proceed.")
             return


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Check if loaders are empty
    if len(train_loader) == 0:
        print("Train loader is empty. This might be due to all samples failing to load or a very small dataset with batch size > num_samples.")
        print(f"Train dataset size: {len(train_dataset)}")
        return
    if len(val_loader) == 0 and len(val_dataset) > 0: # If val_dataset has items but loader is empty
        print("Validation loader is empty, but validation dataset is not. Check batch size vs dataset size.")
        print(f"Validation dataset size: {len(val_dataset)}")
        # It's okay if val_loader is empty if val_dataset is empty, but not if val_dataset has items.


    # Build U-Net model
    model = UNet(n_channels=IMG_CHANNELS, n_classes=1).to(device)
    print(model) # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Lowered learning rate
    criterion = bce_dice_loss  # Use BCE+Dice loss

    # Training loop
    epochs = 60  # Increased epochs
    history = {'loss': [], 'val_loss': [], 'dice_coefficient': [], 'val_dice_coefficient': []}

    print("Starting model training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            dice_coeff = 1 - loss # Since loss is 1 - dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_coeff.item()
            train_pbar.set_postfix({'loss': loss.item(), 'dice': dice_coeff.item()})

        epoch_loss = running_loss / len(train_loader)
        epoch_dice = running_dice / len(train_loader)
        history['loss'].append(epoch_loss)
        history['dice_coefficient'].append(epoch_dice)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_running_dice = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                dice_coeff = 1 - loss

                val_running_loss += loss.item()
                val_running_dice += dice_coeff.item()
                val_pbar.set_postfix({'val_loss': loss.item(), 'val_dice': dice_coeff.item()})
        
        if len(val_loader) > 0:
            epoch_val_loss = val_running_loss / len(val_loader)
            epoch_val_dice = val_running_dice / len(val_loader)
        else: # Handle case where val_loader might be empty (e.g. very small val set)
            epoch_val_loss = 0.0
            epoch_val_dice = 0.0

        history['val_loss'].append(epoch_val_loss)
        history['val_dice_coefficient'].append(epoch_val_dice)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}")

    print("Training finished.")

    # Save model after training
    model_save_path = "model_final_unet2.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if history['val_loss']: plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['dice_coefficient'], label='Training Dice Coefficient')
    if history['val_dice_coefficient']: plt.plot(history['val_dice_coefficient'], label='Validation Dice Coefficient')
    plt.title('Dice Coefficient Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Visualize predictions on a few validation samples
    if len(val_dataset) > 0 and len(val_loader) > 0 : # Ensure val_loader is not empty
        print("Visualizing predictions on validation set...")
        visualize_predictions(model, val_loader, device, num_samples=5)
    else:
        print("Skipping visualization as validation set/loader is empty.")


if __name__ == '__main__':
    # Set num_workers based on OS for DataLoader to avoid issues with multiprocessing on Windows in some environments
    # if os.name == 'nt': # Windows
    #    torch.multiprocessing.freeze_support() # Might be needed if running from script directly
    #    # num_workers = 0 # Often safer on Windows for small datasets/debugging
    # else:
    #    num_workers = 2
    # The num_workers=2 is set directly in DataLoader for now. Adjust if issues arise.
    main()
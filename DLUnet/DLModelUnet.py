import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Suppress sklearn 'zero_division' warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- 1. U-Net Model Definition ---
class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.up3(b)
        # Input tensors to cat must have the same size, but got [H, W] and [H', W']
        # If input size is not divisible by 2^N, feature maps can have size mismatches.
        # A simple fix is to resize `e3` to match `d3`.
        d3 = torch.cat([d3, T.functional.resize(e3, d3.shape[2:])], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, T.functional.resize(e2, d2.shape[2:])], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, T.functional.resize(e1, d1.shape[2:])], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)

# --- 2. Custom Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, manual_label_dir, file_list, transform=None, label_transform=None):
        self.image_dir = image_dir
        self.manual_label_dir = manual_label_dir
        self.file_list = [os.path.splitext(f)[0] for f in file_list]
        self.transform = transform
        self.label_transform = label_transform
        self.image_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG")

    def __len__(self):
        return len(self.file_list)

    def _find_file(self, folder, base_name):
        for ext in self.image_exts:
            path = os.path.join(folder, f"{base_name}{ext}")
            if os.path.exists(path):
                return path
        return None

    def __getitem__(self, idx):
        base_name = self.file_list[idx]
        img_path = self._find_file(self.image_dir, base_name)
        label_path = self._find_file(self.manual_label_dir, base_name)

        if not img_path:
            raise FileNotFoundError(f"Image for base '{base_name}' not found in {self.image_dir}")
        if not label_path:
            raise FileNotFoundError(f"Manual label for base '{base_name}' not found in {self.manual_label_dir}")

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Single-channel grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        # Binarize label: 0 for background, 1 for vein (label==255)
        label_np = np.array(label)
        label_np = (label_np == 255).astype(np.uint8)
        label_tensor = torch.from_numpy(label_np).long()

        return image, label_tensor

# --- 3. Training and Evaluation Functions ---
def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, num_epochs=20, early_stopping_patience=5):
    best_val_f1 = -1.0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate_model(model, val_loader, loss_fn, device, silent=True)
        
        # --- NEW: Step the scheduler with the validation loss ---
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.1e}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            # Only save if F1 score is meaningful
            if best_val_f1 > 0:
                print(f"  -> New best F1-score: {best_val_f1:.4f}. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best F1-score: {best_val_f1:.4f}")
                break
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("Best model parameters restored.")
    return model

def evaluate_model(model, data_loader, loss_fn, device, silent=False):
    model.eval()
    total_loss = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_targets.append(masks.cpu().numpy().flatten())
            all_preds.append(preds.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(data_loader)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    accuracy = np.mean(all_targets == all_preds)
    # Use 'binary' average for binary classification and specify pos_label=1
    precision = precision_score(all_targets, all_preds, pos_label=1, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, pos_label=1, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, pos_label=1, average='binary', zero_division=0)
    
    if not silent:
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 (Vein): {f1:.4f}, Precision (Vein): {precision:.4f}, Recall (Vein): {recall:.4f}")
        
    return avg_loss, accuracy, f1, precision, recall

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    IMAGE_SIZE = 256
    BATCH_SIZE = 64 # Increase batch size if your GPU allows
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 7
    
    # --- Paths ---
    # !!! IMPORTANT: Update these paths to match your folder structure !!!
    image_dir = "images/pictures/"
    manual_label_dir = "images/manual/"
    
    # --- Data Preparation ---
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png'))]
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    n_train = int(0.8 * len(all_files))
    n_val = int(0.1 * len(all_files))
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    # --- Transformations ---
    # CORRECT: Use augmentations ONLY for the training set
    train_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Add more aggressive augmentations
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(30), # Increase rotation angle
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)), # Randomly crop and resize
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # Adjust colors
        T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)), # Apply blur
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # CORRECT: Validation and Test sets should NOT have augmentations, only resizing and normalization
    val_test_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # CORRECT: Mask transform should only resize using NEAREST interpolation
    label_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.NEAREST)
    ])

    # --- Datasets and DataLoaders ---
    train_dataset = SegmentationDataset(image_dir, manual_label_dir, train_files, transform=train_transform, label_transform=label_transform)
    val_dataset = SegmentationDataset(image_dir, manual_label_dir, val_files, transform=val_test_transform, label_transform=label_transform)
    test_dataset = SegmentationDataset(image_dir, manual_label_dir, test_files, transform=val_test_transform, label_transform=label_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images, testing on {len(test_dataset)} images.")

    # --- Model, Loss, and Optimizer ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetSmall(in_channels=3, num_classes=2).to(device)
    
    # CORRECT: The weight for the minority class (vein=1) should be higher.
    # Assuming background (0) is 9x more common than vein (1).
    class_weights = torch.tensor([1.0, 10.0], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # --- Training ---
    model = train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, num_epochs=NUM_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE)
    
    # --- Final Evaluation ---
    print("\n--- Evaluating on Test Set ---")
    evaluate_model(model, test_loader, loss_fn, device)
    
    # --- Save Model ---
    model_save_path = "model_final_unet.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nFinal model saved to {model_save_path}")
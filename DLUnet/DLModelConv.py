import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
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

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        return out

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device='cpu', num_epochs=20, early_stopping_patience=5):
    best_val_loss = float('inf')
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
        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, loss_fn, device, silent=True)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("Best model parameters restored (early stopping).")

def evaluate_model(model, data_loader, loss_fn, device='cpu', silent=False):
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
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    if not silent:
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return avg_loss, accuracy, precision, recall, f1

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, transform=None, mask_transform=None, image_exts=(".jpg", ".jpeg", ".tif", ".tiff", ".png", ".JPG")):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_exts = image_exts
    def __len__(self):
        return len(self.file_list)
    def _find_file(self, base, folder, suffixes):
        for ext in self.image_exts:
            candidate = os.path.join(folder, base + ext)
            if os.path.exists(candidate):
                return candidate
        # Try with suffixes (for mask, e.g. _mask)
        for suf in suffixes:
            for ext in self.image_exts:
                candidate = os.path.join(folder, base + suf + ext)
                if os.path.exists(candidate):
                    return candidate
        return None
    def __getitem__(self, idx):
        base = self.file_list[idx]
        img_path = self._find_file(base, self.image_dir, suffixes=["", "_h", "_g", "_dr"])  # try base, base_h, etc.
        mask_path = self._find_file(base, self.mask_dir, suffixes=["_mask"])
        if img_path is None:
            raise FileNotFoundError(f"Image file for base '{base}' not found in {self.image_dir}")
        if mask_path is None:
            raise FileNotFoundError(f"Mask file for base '{base}' not found in {self.mask_dir}")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)  # binarize if needed
        if self.transform:
            image = self.transform(image)
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, T.Resize):
                        mask = Image.fromarray(mask)
                        mask = t(mask)
                        mask = np.array(mask)
        else:
            image = T.ToTensor()(image)
        mask = torch.from_numpy(mask).long()
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

if __name__ == "__main__":
    # Paths
    image_dir = "images/pictures/"
    mask_dir = "images/mask/"
    # Accept .jpg, .JPG, .jpeg, .tif, .tiff, .png
    all_bases = []
    for f in os.listdir(image_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in ('.jpg', '.jpeg', '.tif', '.tiff', '.png'):
            all_bases.append(name)
    np.random.seed(42)
    np.random.shuffle(all_bases)
    n_train = int(0.8 * len(all_bases))
    n_val = int(0.1 * len(all_bases))
    train_bases = all_bases[:n_train]
    val_bases = all_bases[n_train:n_train+n_val]
    test_bases = all_bases[n_train+n_val:]

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])

    train_dataset = SegmentationDataset(image_dir, mask_dir, train_bases, transform=transform)
    val_dataset = SegmentationDataset(image_dir, mask_dir, val_bases, transform=T.ToTensor())
    test_dataset = SegmentationDataset(image_dir, mask_dir, test_bases, transform=T.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetSmall(in_channels=3, num_classes=2).to(device)
    # Weights for CrossEntropyLoss to handle 1:9 class imbalance.
    # Assumes class 1 is the minority class and class 0 is the majority.
    # The weight for class 0 is 1.0, for class 1 is 9.0.
    # Adjust these weights if your class indexing or distribution is different.
    class_weights = torch.tensor([9.0, 1.0], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")
    train_model(model, train_loader, val_loader, loss_fn, optimizer, device=device, num_epochs=30, early_stopping_patience=5)
    torch.save(model.state_dict(), "model_final_conv.pth")
    print("Model parameters saved to model_final_conv.pth")

    print("Evaluating on test set...")
    evaluate_model(model, test_loader, loss_fn, device=device)

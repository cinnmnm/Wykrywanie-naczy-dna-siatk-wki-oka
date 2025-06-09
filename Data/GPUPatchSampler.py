import torch
from torch.utils.data import Dataset
import numpy as np
import concurrent.futures

class GPUPatchSampler(Dataset):
    def __init__(self, gpu_dataset, patch_size=27, mask_threshold=0.5, balance=False, class_ratio=1.0, precomputed_indices=None, transform=None):
        self.dataset = gpu_dataset # Expects .images, .masks, .labels attributes
        self.patch_size = patch_size
        self.half = self.patch_size // 2
        self.mask_threshold = mask_threshold
        self.balance = balance
        self.class_ratio = class_ratio
        self.transform = transform
        if precomputed_indices is not None:
            self.valid_indices = precomputed_indices
        else:
            self.valid_indices = self._precompute_valid_indices(mask_threshold)
            if self.balance:
                self.valid_indices = self._balance_indices(self.valid_indices, class_ratio)

    def _precompute_valid_indices(self, threshold):
        indices = []
        num_images = self.dataset.masks.shape[0]
        patch_size = self.patch_size
        half = self.half

        def process_one_image(img_idx):
            current_mask_tensor = self.dataset.masks[img_idx]  # [C, H, W] or [H, W]
            if current_mask_tensor.ndim == 3:
                mask_2d = current_mask_tensor[0]  # [H, W]
            elif current_mask_tensor.ndim == 2:
                mask_2d = current_mask_tensor
            else:
                raise ValueError(
                    f"Unsupported mask shape for image {img_idx}: {current_mask_tensor.shape}. "
                    "Expected 2D [H, W] or 3D [C, H, W]."
                )
            mask_2d = mask_2d.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            H, W = mask_2d.shape[-2:]
            if H < patch_size or W < patch_size:
                return []
            patches = torch.nn.functional.unfold(mask_2d, kernel_size=patch_size, stride=1)  # [1, patch_size*patch_size, N]
            valid = (patches > threshold).all(dim=1).squeeze(0)  # [N]
            out_h = H - patch_size + 1
            out_w = W - patch_size + 1
            ys, xs = torch.meshgrid(torch.arange(out_h, device=mask_2d.device), torch.arange(out_w, device=mask_2d.device), indexing='ij')
            centers = torch.stack([ys, xs], dim=-1).reshape(-1, 2)  # [N,2]
            valid_centers = centers[valid]
            result = []
            for center in valid_centers:
                r_center, c_center = int(center[0].item()) + half, int(center[1].item()) + half
                label = self.dataset.labels[img_idx, :, r_center, c_center]
                if label.numel() > 1:
                    label_val = label[0].item()
                else:
                    label_val = label.item()
                label_class = int(label_val > 0.5)
                result.append((img_idx, r_center, c_center, label_class))
            return result

        # Use ThreadPoolExecutor for parallel processing (safe for CUDA context in main thread)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            all_results = list(executor.map(process_one_image, range(num_images)))
        for res in all_results:
            indices.extend(res)
        return indices

    def _balance_indices(self, indices, class_ratio=1.0):
        indices = np.array(indices)
        class0 = indices[indices[:, 3] == 0]
        class1 = indices[indices[:, 3] == 1]
        n1 = len(class1)
        n0 = int(n1 * class_ratio)
        if n0 > len(class0):
            n0 = len(class0)
        np.random.shuffle(class0)
        np.random.shuffle(class1)
        balanced = np.concatenate([class0[:n0], class1])
        np.random.shuffle(balanced)
        return balanced[:, :3].astype(int).tolist()

    def __getitem__(self, idx):
        # Support both 3-tuple and 4-tuple valid_indices (for backward compatibility)
        tup = self.valid_indices[idx]
        if len(tup) == 3:
            img_idx, y_center, x_center = tup
        else:
            img_idx, y_center, x_center, *_ = tup
        img_r_start = y_center - self.half
        img_r_end = y_center + self.half + 1
        img_c_start = x_center - self.half
        img_c_end = x_center + self.half + 1
        image_patch = self.dataset.images[img_idx, :, img_r_start:img_r_end, img_c_start:img_c_end]  # float32, [C,H,W], [0,1]
        label = self.dataset.labels[img_idx, :, y_center, x_center].view(-1)[0].long()  # 0D tensor
        if self.transform:
            image_patch = self.transform(image_patch)
        return image_patch, label

    def __len__(self):
        # This method is required by PyTorch Dataset class for len(dataset) to work.
        return len(self.valid_indices)

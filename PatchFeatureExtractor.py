import numpy as np
import cv2
import csv
from skimage.measure import moments_central

class PatchFeatureExtractor:
    # piksele w wycinku są cechami - 5 x 5 x 3 cech
    def extract_patches(self, images: list, truths: list, fovs: list, patch_size: int = 5) -> tuple[np.ndarray, np.array]:
        if len(images) != len(truths) or len(images) != len(fovs):
            return np.array([]), np.array([])

        h, w = images[0].shape[:2]
        patches = []
        labels = []
        for img, truth, fov in zip(images, truths, fovs):
            for y in range(0, h - patch_size + 1):
                for x in range(0, w - patch_size + 1):
                    patch_mask = fov[y:y+patch_size, x:x+patch_size]
                    if np.any(patch_mask[...,0] == 0):
                        continue
                    patch = img[y:y+patch_size, x:x+patch_size, :]
                    patches.append(patch)
                    center_y = y + patch_size // 2
                    center_x = x + patch_size // 2
                    label = truth[center_y, center_x]
                    labels.append(label)
        return np.array(patches), np.array(labels)
    
    def extract_features(self, patch):
        color_vars = self.color_variance(patch)   # c
        #central_moms = self.central_moments(patch)   # 3
        #hu_moms = self.hu_moments(patch)             # 7

        features = np.concatenate([color_vars])
        return features
    
    # nie działa dla extract_features
    def save_to_csv(self, path: str, patches: np.ndarray, labels: np.ndarray):
        num_features = patches.shape[1] * patches.shape[2] * (patches.shape[3] if patches.ndim == 4 else 1)
        with open(path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [f'feature{i+1}' for i in range(num_features)] + ['label']
            writer.writerow(header)
            for patch, label in zip(patches, labels):
                flat_patch = patch.flatten()
                row = flat_patch.tolist() + [label]
                writer.writerow(row)

    def color_variance(self, patch: np.ndarray) -> np.ndarray:
        # patch: h x w x c
        # Output: c (variance for each channel)
        if patch.ndim != 3:
            print("patch should be of shape: h x w x c")
            return
        return np.var(patch, axis=(0, 1))

    def central_moments(self, patch: np.ndarray) -> np.ndarray:
        # patch: h x w x c
        # Output: 3 values [m[2,0], m[1,1], m[0,2]]
        if patch.ndim != 3:
            print("patch should be of shape: h x w x c")
            return
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        m = moments_central(gray)
        return np.array([m[2, 0], m[1, 1], m[0, 2]])

    def hu_moments(self, patch: np.ndarray) -> np.ndarray:
        # patch: h x w x c
        if patch.ndim != 3:
            raise ValueError("Input must be 3D (h x w x c)")
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        m = cv2.moments(gray)
        hu = cv2.HuMoments(m).flatten()
        return hu

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
    
    def extract_features(self, patches: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # patches: n x h x w x c, labels: n
        color_vars = self.color_variance(patches)  # n x c
        central_moms = self.central_moments(patches)  # n x 3
        hu_moms = self.hu_moments(patches)  # n x 7

        # Concatenate all features and labels column-wise
        features = np.concatenate([color_vars, central_moms, hu_moms], axis=1)
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

    def color_variance(self, patches: np.ndarray) -> np.ndarray:
        # patches: n x h x w x c
        # Output: n x c (variance for each channel per patch)
        if patches.ndim != 4:
            print("patches should be of shape: n x h x w x c")
            return
        return np.var(patches, axis=(1, 2))

    def central_moments(self, patches: np.ndarray) -> np.ndarray:
        # patches: n x h x w x c or n x h x w
        # Output: n x 3 (for each patch: [m[2,0], m[1,1], m[0,2]])
        if patches.ndim != 4:
            print("patches should be of shape: n x h x w x c")
            return
        
        n = patches.shape[0]
        results = []
        for i in range(n):
            patch = patches[i]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            m = moments_central(gray)
            results.append([m[2,0], m[1,1], m[0,2]])
        return np.array(results)

    def hu_moments(self, patches: np.ndarray) -> np.ndarray:
        # patches: n x h x w x c
        if patches.ndim != 4:
            raise ValueError("Input must be 4D (n x h x w x c)")
        n = patches.shape[0]
        results = []
        for i in range(n):
            patch = patches[i]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            m = cv2.moments(gray)
            hu = cv2.HuMoments(m).flatten()
            results.append(hu)
        return np.array(results)

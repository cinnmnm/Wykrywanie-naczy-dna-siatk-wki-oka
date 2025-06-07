import numpy as np
import cv2
import csv
from skimage.measure import moments_central
from scipy.ndimage import convolve

class PatchFeatureExtractor:
    # piksele w wycinku są cechami - 5 x 5 x 3 cech
    def extract_patches(self, images: list, labels: list, masks: list, patch_size: int = 27, patches_per_class: int = 10000) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts a balanced number of patches from each class (1 and 0) per image.
        Returns arrays of patches and their corresponding labels.
        """

        patches = []
        patch_labels = []

        half_patch = patch_size // 2
        kernel = np.ones((patch_size, patch_size), dtype=np.uint8)

        class_patches_per_img = patches_per_class // (len(images) * 2);

        for img, label, mask in zip(images, labels, masks):
            positive_idx = []
            negative_idx = []

            # Convert mask to single channel if needed
            if mask.ndim == 3 and mask.shape[2] > 1:
                mask_gray = (mask[..., 0] > 0).astype(np.uint8)
            else:
                mask_gray = (mask > 0).astype(np.uint8)

            valid_mask = convolve(mask_gray, kernel, mode='constant', cval=0)
            valid_mask = valid_mask == (patch_size * patch_size)

            print("Number of valid patch centers:", np.sum(valid_mask))

            # Ensure label is at least 2D
            if label.ndim == 1:
                raise ValueError("Label array must be at least 2D")
            for y in range(half_patch, label.shape[0] - half_patch):
                for x in range(half_patch, label.shape[1] - half_patch):
                    if valid_mask[y, x]:
                        if label[y, x] == 1:
                            positive_idx.append((y, x))
                        elif label[y, x] == 0:
                            negative_idx.append((y, x))

            np.random.shuffle(positive_idx)
            np.random.shuffle(negative_idx)

            num_pos = min(len(positive_idx), class_patches_per_img)
            num_neg = min(len(negative_idx), class_patches_per_img)

            selected_pos = positive_idx[:num_pos]
            selected_neg = negative_idx[:num_neg]

            for y, x in selected_pos:
                patch = img[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
                patches.append(patch)
                patch_labels.append(1)

            for y, x in selected_neg:
                patch = img[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
                patches.append(patch)
                patch_labels.append(0)

        return np.array(patches), np.array(patch_labels)
    
    def extract_features(self, patch):
        color_vars = self.color_variance(patch)   # c
        central_moms = self.central_moments(patch)   # 3
        hu_moms = self.hu_moments(patch)             # 7

        features = np.concatenate([color_vars, central_moms, hu_moms])
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
        # Output: 3 values for each channel + 3 for gray: [m[2,0], m[1,1], m[0,2]] * (c+1)
        if patch.ndim != 3:
            print("patch should be of shape: h x w x c")
            return
        features = []
        # For each color channel
        for c in range(patch.shape[2]):
            m = moments_central(patch[..., c])
            features.extend([m[2, 0], m[1, 1], m[0, 2]])
        # For grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        m_gray = moments_central(gray)
        features.extend([m_gray[2, 0], m_gray[1, 1], m_gray[0, 2]])
        return np.array(features)

    def hu_moments(self, patch: np.ndarray) -> np.ndarray:
        # patch: h x w x c
        if patch.ndim != 3:
            raise ValueError("Input must be 3D (h x w x c)")
        features = []
        # For each color channel
        for c in range(patch.shape[2]):
            m = cv2.moments(patch[..., c])
            hu = cv2.HuMoments(m).flatten()
            features.extend(hu)
        # For grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        m_gray = cv2.moments(gray)
        hu_gray = cv2.HuMoments(m_gray).flatten()
        features.extend(hu_gray)
        return np.array(features)

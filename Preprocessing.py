import cv2
import numpy as np

class ImagePreprocessing:
    @staticmethod
    def global_contrast_normalization(patches):
        """
        Applies global contrast normalization to a batch of images.
        patches: numpy array of shape (N, H, W) or (N, H, W, C)
        """
        mean = np.mean(patches, axis=(1, 2), keepdims=True)
        sd = np.std(patches, axis=(1, 2), keepdims=True)
        return (patches - mean) / (sd + 1e-8)
    
    @staticmethod
    def zca_whitening(patches, epsilon=1e-5):
        """
        Applies ZCA Whitening to a batch of image patches.
        patches: numpy array of shape (N, H, W) or (N, H, W, C)
        Returns: numpy array of the same shape as input, whitened.
        """
        # Reshape to (N, D)
        orig_shape = patches.shape
        N = orig_shape[0]
        flat_patches = patches.reshape(N, -1)
        
        # Center the data
        mean = np.mean(flat_patches, axis=0, keepdims=True)
        centered = flat_patches - mean

        # Covariance matrix
        sigma = np.cov(centered, rowvar=False)

        # SVD
        U, S, _ = np.linalg.svd(sigma)
        # ZCA matrix
        zca_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T

        # Apply ZCA
        whitened = centered @ zca_matrix.T

        # Reshape back to original
        whitened = whitened.reshape(orig_shape)
        return whitened

    @staticmethod
    def extract_green_channel(images):
        """
        Extracts the green channel from a batch of BGR images.
        images: numpy array of shape (N, H, W, 3)
        Returns: numpy array of shape (N, H, W)
        """
        return images[:, :, :, 1]

    @staticmethod
    def apply_clahe(images, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Applies CLAHE to a batch of grayscale images.
        images: numpy array of shape (N, H, W)
        Returns: numpy array of shape (N, H, W)
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return np.stack([clahe.apply(img) for img in images], axis=0)

    @staticmethod
    def median_filter(images, kernel_size=5):
        """
        Applies median filtering to a batch of images.
        images: numpy array of shape (N, H, W)
        Returns: numpy array of shape (N, H, W)
        """
        return np.stack([cv2.medianBlur(img, kernel_size) for img in images], axis=0)

    @staticmethod
    def normalize(images):
        """
        Normalizes a batch of images to the range [0, 1].
        images: numpy array of shape (N, H, W) or (N, H, W, C)
        """
        images = images.astype(np.float32)
        min_vals = np.min(images, axis=(1, 2), keepdims=True)
        max_vals = np.max(images, axis=(1, 2), keepdims=True)
        return (images - min_vals) / (max_vals - min_vals + 1e-8)

    @staticmethod
    def adaptive_gamma_correction(images, gamma_min=0.5, gamma_max=2.0):
        """
        Applies adaptive gamma correction to a batch of images.
        images: numpy array of shape (N, H, W)
        Returns: numpy array of shape (N, H, W)
        """
        corrected_images = []
        for img in images:
            mean_intensity = np.mean(img)
            gamma = gamma_max - (gamma_max - gamma_min) * mean_intensity / 255.0
            img_norm = img / 255.0
            corrected = np.power(img_norm, gamma)
            corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)
            corrected_images.append(corrected)
        return np.stack(corrected_images, axis=0)

    @staticmethod
    def resize_and_normalize(images, size=(512, 512)):
        """
        Resizes and normalizes a batch of images using downsampling with minimal information loss.
        images: numpy array of shape (N, H, W) or (N, H, W, C)
        Returns: numpy array of shape (N, size[0], size[1]) or (N, size[0], size[1], C)
        """
        # Use INTER_LANCZOS4 for high-quality downsampling
        resized = np.stack([
            cv2.resize(
                img.astype(np.uint8) if img.dtype != np.uint8 else img,
                size,
                interpolation=cv2.INTER_LANCZOS4
            ) if img.ndim == 2 or img.ndim == 3 else img
            for img in images
        ], axis=0)
        normalized = ImagePreprocessing.normalize(resized)
        return normalized

import cv2
import numpy as np

class ImagePreprocessing:
    @staticmethod
    def global_contrast_normalization(patch):
        """
        Applies global contrast normalization to a single image.
        patch: numpy array of shape (H, W) or (H, W, C)
        """
        mean = np.mean(patch)
        sd = np.std(patch)
        return (patch - mean) / (sd + 1e-8)

    @staticmethod
    def histogram_normalization(image):
        """
        Normalizes the histogram of a grayscale image.
        image: numpy array of shape (H, W)
        Returns: numpy array of shape (H, W)
        """
        return cv2.equalizeHist(image)

    @staticmethod
    def extract_green_channel(image):
        """
        Extracts the green channel from a BGR image.
        image: numpy array of shape (H, W, 3)
        Returns: numpy array of shape (H, W)
        """
        return image[:, :, 1]
    
    @staticmethod
    def grayscale(image):
        """
        Converts a BGR image to grayscale.
        image: numpy array of shape (H, W, 3)
        Returns: numpy array of shape (H, W)
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Applies CLAHE to a grayscale image.
        image: numpy array of shape (H, W)
        Returns: numpy array of shape (H, W)
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    @staticmethod
    def median_filter(image, kernel_size=5):
        """
        Applies median filtering to an image.
        image: numpy array of shape (H, W)
        Returns: numpy array of shape (H, W)
        """
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def normalize(image):
        """
        Normalizes an image to the range [0, 1].
        image: numpy array of shape (H, W) or (H, W, C)
        """
        return ImagePreprocessing.normalize(image) * 255

    @staticmethod
    def adaptive_gamma_correction(image, gamma_min=0.5, gamma_max=2.0):
        """
        Applies adaptive gamma correction to an image.
        image: numpy array of shape (H, W)
        Returns: numpy array of shape (H, W)
        """
        mean_intensity = np.mean(image)
        gamma = gamma_max - (gamma_max - gamma_min) * mean_intensity / 255.0
        img_norm = image / 255.0
        corrected = np.power(img_norm, gamma)
        corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)
        return corrected

    @staticmethod
    def resize_and_normalize(image, size=(512, 512)):
        """
        Resizes and normalizes an image using downsampling with minimal information loss.
        image: numpy array of shape (H, W) or (H, W, C)
        Returns: numpy array of shape (size[0], size[1]) or (size[0], size[1], C)
        """
        img_resized = cv2.resize(
            image.astype(np.uint8) if image.dtype != np.uint8 else image,
            size,
            interpolation=cv2.INTER_LANCZOS4
        )
        normalized = ImagePreprocessing.normalize(img_resized)
        return normalized

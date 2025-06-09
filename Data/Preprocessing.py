import cv2
import numpy as np
from typing import Tuple
from PIL import Image
import numpy as np

class ImagePreprocessing:
    @staticmethod
    def global_contrast_normalization(patch: np.ndarray) -> np.ndarray:
        """
        Applies global contrast normalization to a single image.
        patch: numpy array of shape (H, W) or (H, W, C)
        Returns: numpy array, float, with normalized contrast.
        """
        if not isinstance(patch, np.ndarray):
            raise TypeError("Input patch must be a numpy array.")
        if patch.size == 0:
            raise ValueError("Input patch cannot be empty.")

        mean = np.mean(patch)
        sd = np.std(patch)
        return (patch - mean) / (sd + 1e-8)

    @staticmethod
    def histogram_normalization(image: np.ndarray) -> np.ndarray:
        """
        Normalizes the histogram of a grayscale image.
        image: numpy array of shape (H, W), dtype=np.uint8
        Returns: numpy array of shape (H, W), dtype=np.uint8
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D (grayscale). Got {image.ndim} dimensions.")
        if image.dtype != np.uint8:
            raise TypeError(f"Image dtype must be np.uint8 for histogram equalization. Got {image.dtype}.")
        
        return cv2.equalizeHist(image)

    @staticmethod
    def extract_green_channel(image: np.ndarray) -> np.ndarray:
        """
        Extracts the green channel from a BGR image.
        image: numpy array of shape (H, W, 3)
        Returns: numpy array of shape (H, W) representing the green channel.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be 3-channel (H, W, 3). Got shape {image.shape}.")
        
        return image[:, :, 1]
    
    @staticmethod
    def grayscale(image: np.ndarray) -> np.ndarray:
        """
        Converts a BGR image to grayscale.
        image: numpy array of shape (H, W, 3)
        Returns: numpy array of shape (H, W)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be 3-channel (H, W, 3) for BGR to Gray conversion. Got shape {image.shape}.")

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Applies CLAHE to a grayscale or multi-channel image (channel-wise).
        Image channels must be of dtype np.uint8.
        image: numpy array of shape (H, W) or (H, W, C), dtype=np.uint8
        Returns: numpy array of shape (H, W) or (H, W, C), dtype=np.uint8
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")
        if image.dtype != np.uint8:
            raise TypeError(f"Image dtype must be np.uint8 for CLAHE. Got {image.dtype}.")
        if not isinstance(clip_limit, float):
            raise TypeError(f"clip_limit must be a float. Got {type(clip_limit)}.")
        if not (isinstance(tile_grid_size, tuple) and len(tile_grid_size) == 2 and
                all(isinstance(x, int) and x > 0 for x in tile_grid_size)):
            raise TypeError("tile_grid_size must be a tuple of two positive integers.")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if image.ndim == 3:
            channels = [clahe.apply(image[:, :, i]) for i in range(image.shape[2])]
            return cv2.merge(channels)
        elif image.ndim == 2:
            return clahe.apply(image)
        else:
            raise ValueError(f"Image must be 2D or 3D. Got {image.ndim} dimensions.")


    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Applies median filtering to an image.
        image: numpy array of shape (H, W) or (H, W, C)
        Returns: numpy array of the same shape and dtype as input.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")
        if not isinstance(kernel_size, int):
            raise TypeError(f"kernel_size must be an integer. Got {type(kernel_size)}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd positive integer.")
            
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalizes an image to the range [0, 1] (float).
        image: numpy array of shape (H, W) or (H, W, C)
        Returns: numpy array of the same shape, dtype=np.float32, with values in [0, 1].
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")

        img_float = image.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        
        if max_val == min_val:
            return np.zeros_like(img_float) 
        
        normalized_image = (img_float - min_val) / (max_val - min_val + 1e-8) 
        return normalized_image

    @staticmethod
    def adaptive_gamma_correction(image: np.ndarray, gamma_min: float = 0.5, gamma_max: float = 2.0) -> np.ndarray:
        """
        Applies adaptive gamma correction to an image.
        Assumes input image is in range [0, 255].
        image: numpy array of shape (H, W), typically np.uint8
        Returns: numpy array of shape (H, W), dtype=np.uint8
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")
        if image.ndim != 2:
             raise ValueError(f"Image must be 2D for adaptive gamma correction. Got {image.ndim} dimensions.")
        if not isinstance(gamma_min, float) or not isinstance(gamma_max, float):
            raise TypeError("gamma_min and gamma_max must be floats.")
        if gamma_min <= 0 or gamma_max <= 0:
            raise ValueError("gamma_min and gamma_max must be positive.")
        if gamma_min > gamma_max:
            raise ValueError("gamma_min cannot be greater than gamma_max.")

        mean_intensity = np.mean(image)
        gamma = gamma_max - (gamma_max - gamma_min) * np.clip(mean_intensity, 0, 255) / 255.0
        
        img_norm = image.astype(np.float32) / 255.0 
        corrected = np.power(img_norm, gamma)
        
        corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
        return corrected

    @staticmethod
    def resize_and_normalize(image: np.ndarray, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Resizes an image and then normalizes it to the range [0, 1] (float).
        image: numpy array of shape (H, W) or (H, W, C)
        size: tuple (width, height) for resizing.
        Returns: numpy array of shape (size[1], size[0]) or (size[1], size[0], C), 
                 dtype=np.float32, with values in [0, 1].
                 Note: cv2.resize takes (width, height), so output shape is (height, width, C).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        if image.size == 0:
            raise ValueError("Input image cannot be empty.")
        if not (isinstance(size, tuple) and len(size) == 2 and
                all(isinstance(x, int) and x > 0 for x in size)):
            raise TypeError("size must be a tuple of two positive integers (width, height).")

        img_resized = cv2.resize(
            image, 
            size,  
            interpolation=cv2.INTER_LANCZOS4
        )
        
        normalized = ImagePreprocessing.normalize(img_resized)
        return normalized

    @staticmethod
    def preprocess_image(img, scale_shape, color_adjustment=True, normalization=True):
        if isinstance(img, str):
            img = Image.open(img)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        img = img.resize(scale_shape[::-1], resample=Image.Resampling.BILINEAR)
        np_img = np.array(img)
        if color_adjustment:
            np_img = ImagePreprocessing.global_contrast_normalization(np_img)
            np_img = np.clip((np_img - np_img.min()) / (np_img.max() - np_img.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
        if normalization:
            np_img = np_img.astype(np.float32) / 255.0
        return np_img  # Always return numpy array

    @staticmethod
    def preprocess_mask_or_label(mask, scale_shape):
        if isinstance(mask, str):
            mask = Image.open(mask)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        mask = mask.convert('L')
        mask = mask.resize(scale_shape[::-1], resample=Image.Resampling.NEAREST)
        arr = np.array(mask)
        arr = (arr > 0).astype(np.int64)
        return arr  # Always return numpy array

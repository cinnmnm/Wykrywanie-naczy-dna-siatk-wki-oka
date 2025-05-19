import cv2
import numpy as np
from skimage.morphology import skeletonize

class ImagePostprocessing:
    @staticmethod
    def adaptive_binarization(image, block_size=11, C=2):
        # Ensure image is single-channel and uint8
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

    @staticmethod
    def morphological_opening(image, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def morphological_closing(image, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def skeletonize_image(image):
        closed_bool = image.astype(bool)
        skeleton = skeletonize(closed_bool).astype(np.uint8) * 255
        return skeleton

    @staticmethod
    def filter_small_objects(image, min_size=50):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image)
        filtered = np.zeros_like(image)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered[labels == i] = 255
        return filtered

    @staticmethod
    def smooth_contours(image, ksize=3):
        return cv2.medianBlur(image, ksize)
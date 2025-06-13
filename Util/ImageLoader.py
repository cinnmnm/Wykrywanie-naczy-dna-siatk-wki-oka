import numpy as np
import cv2

class ImageLoader:
    @staticmethod
    def load_images(image_paths: list[str], BGRtoRGB: bool = False) -> list[np.ndarray]:
        images = []
        for path in image_paths:
            if path.lower().endswith(('.jpg', '.tif')):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Read image in BGR format
                if img is not None:
                    if BGRtoRGB:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    images.append(img.astype(np.uint8))
        return images
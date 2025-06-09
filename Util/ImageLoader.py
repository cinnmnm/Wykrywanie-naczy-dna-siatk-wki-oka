import numpy as np
import cv2

class ImageLoader:
    @staticmethod
    def load_images(image_paths: list[str]) -> list[np.ndarray]:
        images = []
        for path in image_paths:
            if path.lower().endswith(('.jpg', '.tif')):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                img = np.array(img.astype(np.uint8))
                if img is not None:
                    images.append(img)
        return images
from Preprocessing import ImagePreprocessing
from Processing import FrangiFilter2D
from Postprocessing import ImagePostprocessing
import numpy as np

class FilterSegmentation:
    @staticmethod
    def run(image: np.ndarray) -> np.ndarray:
        #print("Input image type:", type(image), "shape:", getattr(image, "shape", None))

        green = ImagePreprocessing.extract_green_channel(image)
        #print("Green channel shape:", getattr(green, "shape", None))

        gray = ImagePreprocessing.grayscale(image)
        clahe = ImagePreprocessing.apply_clahe(gray)
        median_filtered = ImagePreprocessing.median_filter(clahe)
        frangi_result = FrangiFilter2D.apply(median_filtered)
        binary = ImagePostprocessing.adaptive_binarization(frangi_result)
        opened = ImagePostprocessing.morphological_opening(binary)
        closed = ImagePostprocessing.morphological_closing(opened)

        #print("Final closed type:", type(closed), "shape:", getattr(closed, "shape", None))

        return closed

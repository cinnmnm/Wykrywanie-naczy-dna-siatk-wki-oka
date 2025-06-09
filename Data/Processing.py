from skimage.filters import frangi
import numpy as np

class FrangiFilter2D:
    @staticmethod
    def apply(image: np.ndarray, scale_range: tuple[int, int] = (1, 10), scale_step: int = 2, beta: float = 0.5) -> np.ndarray:
        """
        Apply the Frangi filter to a 2D image.

        Parameters:
            image (ndarray): 2D input image.
            scale_range (tuple): The range of sigmas for the filter.
            scale_step (int): Step size between sigmas.
            beta1 (float): Frangi correction constant.

        Returns:
            ndarray: Filtered image.
        """
        return frangi(
            image,
            sigmas=range(scale_range[0], scale_range[1], scale_step),
            beta=beta
        )
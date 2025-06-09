from typing import List, Optional
import matplotlib
import matplotlib.axes
import numpy as np

class Visualisation:
    @staticmethod
    def create_diffmap(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """
        Create a difference map between the reference and predicted images.
        The difference is computed as the absolute difference between the two images.
        
        :param reference: The reference image (ground truth).
        :param predicted: The predicted image.
        :return: A difference map as a numpy array.
        """
        if reference.shape != predicted.shape:
            raise ValueError("Reference and predicted images must have the same shape.")
        
        if reference.ndim != 2:
            raise ValueError("Reference and predicted images must be 2D arrays.")

        BLACK = [0, 0, 0]  
        RED = [255, 0, 0]     
        BLUE = [0, 0, 255]
        WHITE = [255, 255, 255]

        height, width = reference.shape
        diff_map = np.full((height, width, 3), BLACK, dtype=np.uint8)

        ref_present = reference > 0
        pred_present = predicted > 0

        diff_map[ref_present & pred_present] = WHITE
        diff_map[ref_present & (~pred_present)] = RED
        diff_map[(~ref_present) & pred_present] = BLUE

        return diff_map
    
    @staticmethod
    def plot_comparison_row(
        original_image: np.ndarray,
        ground_truth_label: np.ndarray,
        model_prediction: np.ndarray,
        axes_row: Optional[List[matplotlib.axes.Axes]] = None,
        original_title: str = "Original Image (Resized + GCN)",
        label_title: str = "Ground Truth Label (Resized)",
        prediction_title: str = "Model Prediction",
        diff_map_title: str = "Diff map"
    ):
        """
        Displays a row of four images: original, ground truth, prediction, and difference map.
        If axes_row is provided (a list/array of 4 matplotlib Axes objects), plots on these axes.
        Otherwise, creates a new figure for a single row and displays it.

        Requires `matplotlib.pyplot` to be imported (e.g., as plt).
        Requires `from typing import Optional, List` for the type hint of `axes_row`.

        :param original_image: The original image (can be 2D grayscale or 3D RGB).
        :param ground_truth_label: The ground truth segmentation map (2D).
        :param model_prediction: The predicted segmentation map (2D).
        :param axes_row: Optional. A list or array of 4 matplotlib.axes.Axes objects to plot on.
                         If None, a new figure is created, and subplots are made for a single row,
                         then `plt.show()` is called.
                         Example for plotting multiple rows:
                           import matplotlib.pyplot as plt
                           from typing import Optional, List # Add to top of your file
                           # num_rows = 3
                           # fig, axes = plt.subplots(num_rows, 4, figsize=(18, 6 * num_rows))
                           # for i in range(num_rows):
                           #     # Replace with your actual data loading
                           #     img, gt, pred = my_data_source[i] 
                           #     Visualisation.plot_comparison_row(img, gt, pred, axes_row=axes[i])
                           # fig.tight_layout() # Or plt.tight_layout()
                           # plt.show()
        :param original_title: Title for the original image subplot.
        :param label_title: Title for the ground truth label subplot.
        :param prediction_title: Title for the model prediction subplot.
        :param diff_map_title: Title for the difference map subplot.
        """
        import matplotlib.pyplot as plt

        # Ensure ground_truth_label and model_prediction are 2D for cmap='gray' and create_diffmap
        if ground_truth_label.ndim != 2:
            raise ValueError("Ground truth label must be a 2D array.")
        if model_prediction.ndim != 2:
            raise ValueError("Model prediction must be a 2D array.")

        diff_map = Visualisation.create_diffmap(ground_truth_label, model_prediction)

        if axes_row is None:
            # Create a new figure and plot a single row
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 4, 1)
            plt.imshow(original_image)
            plt.title(original_title)
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(ground_truth_label, cmap='gray')
            plt.title(label_title)
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(model_prediction, cmap='gray')
            plt.title(prediction_title)
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(diff_map)
            plt.title(diff_map_title)
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            # Plot on the provided axes_row
            if not (hasattr(axes_row, '__len__') and len(axes_row) == 4):
                raise ValueError(
                    "axes_row must be a list-like object containing 4 matplotlib.axes.Axes objects."
                )
            
            # Further check if elements are Axes (optional, but good for robustness)
            # for ax_obj in axes_row:
            #    if not isinstance(ax_obj, plt.Axes): # Requires plt to be matplotlib.pyplot
            #        raise ValueError("Each element in axes_row must be a matplotlib.axes.Axes object.")

            ax1, ax2, ax3, ax4 = axes_row[0], axes_row[1], axes_row[2], axes_row[3]

            ax1.imshow(original_image)
            ax1.set_title(original_title)
            ax1.axis('off')

            ax2.imshow(ground_truth_label, cmap='gray')
            ax2.set_title(label_title)
            ax2.axis('off')

            ax3.imshow(model_prediction, cmap='gray')
            ax3.set_title(prediction_title)
            ax3.axis('off')

            ax4.imshow(diff_map)
            ax4.set_title(diff_map_title)
            ax4.axis('off')
            # If axes_row is provided, the caller is responsible for plt.tight_layout() and plt.show()
            # on the main figure.
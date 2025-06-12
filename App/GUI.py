import cv2
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
from App.Controller import Controller
from Data.Preprocessing import ImagePreprocessing
from Util.ImageLoader import ImageLoader
import io
import numpy as np
import os
from Util.Visualisation import Visualisation
from Util.Evaluate import Evaluate

class GUI:
    def __init__(self, controller: Controller):
        self.controller = controller
        # Left panel widgets
        self.file_path = widgets.FileUpload(
            accept='image/*',
            multiple=False,
            description='Choose Image',
            layout=widgets.Layout(width='95%')
        )

        self.prosty_model = widgets.Checkbox(
            value=False,
            description='Filtrowanie'
        )
        self.ml_model = widgets.Checkbox(
            value=False,
            description='ML Model'
        )
        self.dl_model = widgets.Checkbox(
            value=False,
            description='DL Model'
        )

        self.start_button = widgets.Button(
            description='Start',
            button_style='success'
        )

        # Right panel: Output for image display
        self.image_output = widgets.Output(layout=widgets.Layout(border='1px solid gray', min_height='300px'))

        self.start_button.on_click(self.on_start_clicked)

        # Layout
        self.left_panel = widgets.VBox([
            self.file_path,
            self.prosty_model,
            self.ml_model,
            self.dl_model,
            self.start_button
        ], layout=widgets.Layout(width='40%', border='1px solid lightgray', padding='10px'))

        self.right_panel = widgets.VBox([
            widgets.Label("Image Display:"),
            self.image_output
        ], layout=widgets.Layout(width='60%', border='1px solid lightgray', padding='10px'))

        self.ui = widgets.HBox(
            [self.left_panel, self.right_panel],
            layout=widgets.Layout(width='100%')
        )
        self.left_panel.layout.width = '30%'
        self.right_panel.layout.width = '70%'

    def on_start_clicked(self, b):
        with self.image_output:
            clear_output()
            if not self.file_path.value:
                print("No file uploaded.")
                return
            
            if not self.prosty_model.value and not self.ml_model.value and not self.dl_model.value:
                print("No model picked.")
                return

            # Get uploaded file content and name
            upload = next(iter(self.file_path.value))
            name = upload['name']

            # Prepare paths for the uploaded image, manual, and mask
            image_dir = "images"
            image_path = os.path.join(image_dir, "pictures", name)
            manual_path = os.path.join(
            image_dir, "manual",
            os.path.basename(name).replace(".JPG", ".tif").replace(".jpg", ".tif")
            )
            mask_path = os.path.join(
            image_dir, "mask",
            os.path.basename(name).replace(".JPG", "_mask.tif").replace(".jpg", "_mask.tif")
            )

            # Check if files exist
            if not os.path.exists(image_path):
                print(f"File {name} does not exist in images/pictures/.")
                return
            if not os.path.exists(manual_path):
                print(f"Manual for {name} does not exist in images/manual/.")
                return
            if not os.path.exists(mask_path):
                print(f"Mask for {name} does not exist in images/mask/.")
                return

            # Load images using ImageLoader
            try:
                images = ImageLoader.load_images([image_path], BGRtoRGB=True)
                manuals = ImageLoader.load_images([manual_path])
                masks = ImageLoader.load_images([mask_path])
                if not images or not manuals or not masks:
                    print("Could not load image, manual, or mask using ImageLoader.")
                    return
                img_cv = images[0]
                manual_cv = manuals[0]
                mask_cv = masks[0]
            except Exception as e:
                print(f"Could not load image, manual, or mask using ImageLoader: {e}")
                return

            
            resized_img = ImagePreprocessing.resize_and_normalize(img_cv)
            resized_img = (resized_img > 0).astype(np.uint8)

            resized_true = ImagePreprocessing.resize_and_normalize(manual_cv)
            resized_true = (resized_true > 0).astype(np.uint8)

            resized_mask = ImagePreprocessing.resize_and_normalize(mask_cv)
            resized_mask = (resized_mask > 0).astype(np.uint8)
            resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

            # Collect results from all selected models
            results = []
            try:
                if self.prosty_model.value:
                    result = self.controller.run_filter(img_cv)
                    results.append(('Filtrowanie', result))
                if self.ml_model.value:
                    result = self.controller.run_ml(img_cv)
                    results.append(('ML Model', result))
                if self.dl_model.value:
                    result = self.controller.run_dl(img_cv)
                    results.append(('DL Model', result))
                if not results:
                    print("No model selected.")
                    return
            except Exception as e:
                print(f"Error running selected model: {e}")
                return

            # Display original, manual, mask, and all processed results
            original_img_widget = self._np_to_widget_image(img_cv)
            manual_img_widget = self._np_to_widget_image(manual_cv)
            mask_img_widget = self._np_to_widget_image(mask_cv)

            import matplotlib.pyplot as plt

            # For each model result, display a row using Visualisation.plot_comparison_row
            for label, result in results:
                # If result is a list or tuple, get the first element (assuming that's the image)
                if isinstance(result, (list, tuple)):
                    result_img = result[0]
                else:
                    result_img = result

                # Prepare a matplotlib figure for each model
                fig, axes = plt.subplots(1, 4, figsize=(18, 4))
                Visualisation.plot_comparison_row(
                    original_image=resized_img,
                    ground_truth_label=resized_true,
                    model_prediction=result_img,
                    axes_row=axes,
                    original_title="Original Image",
                    label_title="Manual",
                    prediction_title=f"Prediction ({label})",
                    diff_map_title="Diff map"
                )
                fig.suptitle(f"Results for {label}", fontsize=16)
                plt.tight_layout()
                plt.show()

                # TODO można najpierw obliczyć cm i później z niej skorzystać zamiast obliczać na nowo
                accuracy = Evaluate.accuracy(resized_true, result_img, resized_mask)
                sensitivity = Evaluate.sensitivity(resized_true, result_img, resized_mask)
                specificity = Evaluate.specificity(resized_true, result_img, resized_mask)

                print(f"Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}")
                Evaluate.print_confusion_matrix(resized_true, result_img, resized_mask)

    def _np_to_widget_image(self, arr):
        if arr.ndim == 2:
            mode = 'L'
            img = Image.fromarray(arr.astype(np.uint8), mode)
        else:
            # Convert BGR to RGB if needed
            if arr.shape[2] == 3:
                arr = arr[..., ::-1]  # Swap channels
            mode = 'RGB'
            img = Image.fromarray(arr.astype(np.uint8), mode)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return widgets.Image(value=buf.getvalue(), format='png')


    def init(self):
        display(self.ui)
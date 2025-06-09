import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
from App.Controller import Controller
from Util.ImageLoader import ImageLoader
import io
import numpy as np
import os

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
            description='Prosty Model'
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

            # Get uploaded file content and name
            upload = next(iter(self.file_path.value))
            name = upload['name']

            # Extract image path from FileUpload (simulate as if uploaded file is in images/pictures/)
            image_dir = "images/pictures/"
            image_path = os.path.join(image_dir, name)

            # Check if file exists in the expected directory
            if not os.path.exists(image_path):
                print(f"File {name} does not exist in images/pictures/.")
                return

            # Load image using ImageLoader
            try:
                images = ImageLoader.load_images([image_path])
                if not images:
                    print("Could not load image using ImageLoader.")
                    return
                img_cv = images[0]
            except Exception as e:
                print(f"Could not load image using ImageLoader: {e}")
                return

            # Pass image to controller
            try:
                result = self.controller.run_filter(img_cv)
                # If result is a list or tuple, get the first element (assuming that's the image)
                if isinstance(result, (list, tuple)):
                    result_img = result[0]
                else:
                    result_img = result
            except Exception as e:
                print(f"Error running filter: {e}")
                return

            # Display original and result in the right panel's image_output
            original_img_widget = self._np_to_widget_image(img_cv)
            processed_img_widget = self._np_to_widget_image(result_img)
            display(widgets.HBox([
                widgets.VBox([widgets.Label("Original Image:"), original_img_widget]),
                widgets.VBox([widgets.Label("Processed Image:"), processed_img_widget])
            ]))
    
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
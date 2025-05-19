import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image

class GUI:
    def __init__(self):
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

        self.ui = widgets.HBox([self.left_panel, self.right_panel], layout=widgets.Layout(width='100%'))

    def on_start_clicked(self, b):
        with self.image_output:
            clear_output()
            print("start")
            path = self.file_path.value.strip()
            if path:
                try:
                    img = Image.open(path)
                    display(img)
                except Exception as e:
                    print(f"Could not open image: {e}")

    def init(self):
        display(self.ui)
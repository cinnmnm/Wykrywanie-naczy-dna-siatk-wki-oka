import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim
from DLModel import DLModel, train_model, evaluate_model
import torch
import numpy as np
from torchvision import transforms as T
import os
from PIL import Image

class DLModelWrapper:
    def __init__(self, input_size: int, num_epochs: int = 10, learning_rate: float = 0.001, batch_size: int = 32, device: str = 'cpu'):
        self.model = DLModel()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        #self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss() # Assuming classification task
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Load the model if a path is provided and the file exists
        model_path = 'model_final.pth'
        if os.path.exists(model_path):
            try:
                # Load the state dict, mapping to the correct device
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Initializing with new model parameters instead.")
        else:
            print(f"Model file {model_path} not found. Initializing with new model parameters.")

        self.model.to(self.device) # Ensure model is on the correct device after loading or init
        self.loss_fn = nn.CrossEntropyLoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if len(X_train.shape) == 3: # (N, H, W)
            X_train = np.expand_dims(X_train, axis=1) # (N, 1, H, W)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        train_model(self.model, train_loader, self.loss_fn, self.optimizer, self.num_epochs, self.device)

    def predict(self, image_param: np.ndarray) -> np.ndarray:
        """
        Generates a prediction map for a single input image using a sliding window of patches.
        The prediction for each patch (assumed to be centered on a pixel) is assigned 
        to that corresponding pixel in the output map.

        Args:
            image_param (np.ndarray): Input image as a NumPy array.
                                Expected shapes: (H, W) for grayscale, or (H, W, C) for color.
                                Pixel values can be uint8 [0, 255] or float32 [0, 1].
                                If float32 and outside [0,1] (e.g. [0,255]), it will be scaled.

        Returns:
            np.ndarray: A 2D map of shape (H, W) containing predicted class labels (int64).
        """
        self.model.eval()
        
        # 1. Preprocess image_param: ensure it's (H, W, 3) as DLModel expects 3 channels.
        #    Also, ensure data type and range are suitable for T.ToTensor().
        current_image = image_param.copy() 

        if current_image.ndim == 2: # Grayscale (H, W)
            current_image = np.stack([current_image]*3, axis=-1) # -> (H, W, 3)
        elif current_image.ndim == 3:
            if current_image.shape[-1] == 1: # Grayscale with channel dim (H, W, 1)
                current_image = np.concatenate([current_image]*3, axis=-1) # -> (H, W, 3)
            elif current_image.shape[-1] == 4: # RGBA (H, W, 4)
                current_image = current_image[:, :, :3] # Drop alpha -> (H, W, 3)
            elif current_image.shape[-1] == 3: # RGB (H, W, 3)
                pass # Already in correct format
            else:
                raise ValueError(
                    f"Input image has {current_image.shape[-1]} channels. Expected 1, 3, or 4 channels."
                )
        else:
            raise ValueError("Input image must be 2D (H, W) or 3D (H, W, C).")

        # Ensure image data is float32 and in [0, 1] range if not uint8
        if current_image.dtype != np.uint8:
            current_image = current_image.astype(np.float32)

        if current_image.dtype == np.float32: # If float, ensure range [0,1]
            if current_image.min() < 0.0 or current_image.max() > 1.0: # Check if not already in [0,1]
                if current_image.max() > 1.0: # Common case: float image in [0, 255] range
                    current_image = current_image / 255.0
                current_image = np.clip(current_image, 0.0, 1.0) # Clip to ensure [0,1]
        # If uint8, T.ToTensor() will handle scaling to [0,1] float tensor.


        H, W, _ = current_image.shape
        # self.input_size is assumed to be the patch_size n, set during DLModelWrapper init.
        patch_size = 27
        if patch_size is None or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("Patch size (self.input_size) is not properly configured in DLModelWrapper.")

        output_map = np.zeros((H, W), dtype=np.int64)

        # Transform for patches: HWC NumPy/PIL to CHW Tensor
        patch_transform = T.Compose([
            T.ToTensor() 
        ])

        # Pad the image to allow extracting patches centered at edge/corner pixels
        pad_amount = patch_size // 2
        padded_image = np.pad(current_image, 
                              ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 
                              mode='reflect')

        patches_batch = []
        # Stores (row, col) in original image for which the prediction is made
        patch_center_coords_in_original = [] 

        for r_orig in range(H): # Iterate over each pixel of the original image
            for c_orig in range(W):
                # Top-left corner of the patch in the *padded* image.
                # This patch is centered at (r_orig, c_orig) of the original image.
                start_r_in_padded = r_orig
                start_c_in_padded = c_orig
                
                patch = padded_image[start_r_in_padded : start_r_in_padded + patch_size, 
                                     start_c_in_padded : start_c_in_padded + patch_size, :]
                
                transformed_patch = patch_transform(patch) # (C, patch_size, patch_size) tensor
                patches_batch.append(transformed_patch)
                patch_center_coords_in_original.append((r_orig, c_orig))

                if len(patches_batch) == self.batch_size:
                    patches_tensor = torch.stack(patches_batch).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(patches_tensor) # (batch_size, num_classes)
                        _, predicted_classes = torch.max(outputs, 1) # (batch_size)
                    
                    predicted_classes_np = predicted_classes.cpu().numpy() # (batch_size)
                    
                    for i in range(len(predicted_classes_np)):
                        coord_r, coord_c = patch_center_coords_in_original[i]
                        output_map[coord_r, coord_c] = predicted_classes_np[i]
                    
                    patches_batch = []
                    patch_center_coords_in_original = []

        # Process any remaining patches in the last batch
        if len(patches_batch) > 0:
            patches_tensor = torch.stack(patches_batch).to(self.device)
            with torch.no_grad():
                outputs = self.model(patches_tensor)
                _, predicted_classes = torch.max(outputs, 1)
            
            predicted_classes_np = predicted_classes.cpu().numpy()
            
            for i in range(len(predicted_classes_np)):
                coord_r, coord_c = patch_center_coords_in_original[i]
                output_map[coord_r, coord_c] = predicted_classes_np[i]
        
        return output_map

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        # Ensure X_test is in the format (N, C, H, W)
        if len(X_test.shape) == 3: # (N, H, W)
            X_test = np.expand_dims(X_test, axis=1) # (N, 1, H, W)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Assuming evaluate_model returns (loss, accuracy) and we want to return accuracy
        loss, accuracy = evaluate_model(self.model, test_loader, self.loss_fn, self.device)
        return accuracy
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np # Ensure numpy is imported if not already at the top level of the file

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Configuration ---
    # This should match the patch size 'n' used during training (e.g., 27 from DLModel.py)
    # DLModelWrapper.predict currently hardcodes patch_size=27.
    patch_input_size = 27 
    image_filename = "01_dr.tif"  # Example filename, user should ensure this exists
    image_base_dir = "images"     # Consistent with DLModel.py
    image_path = os.path.join(image_base_dir, image_filename)

    # --- Instantiate the Wrapper ---
    # The model 'model_final.pth' will be loaded by the wrapper if it exists.
    print("Initializing DLModelWrapper...")
    # batch_size for prediction can be tuned based on available VRAM/memory
    wrapper = DLModelWrapper(input_size=patch_input_size, device=device, batch_size=64)
    print("DLModelWrapper initialized.")

    # --- Load the image ---
    print(f"Attempting to load image: {image_path}")
    img_np = None
    original_image_for_plot = None
    try:
        if not os.path.exists(image_path):
            # Fallback: Try common image extensions if .tif is not found or specified
            found_img = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                potential_path = os.path.join(image_base_dir, "01_dr" + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    print(f"Found image as: {image_path}")
                    found_img = True
                    break
            if not found_img:
                 raise FileNotFoundError(f"Image '01_dr' with common extensions not found in '{image_base_dir}'. Tried .tif, .png, .jpg, .jpeg, .bmp.")

        img = Image.open(image_path)
        img_np = np.array(img)
        original_image_for_plot = img_np.copy() # Keep a copy for plotting
        print(f"Image loaded successfully. Shape: {img_np.shape}, dtype: {img_np.dtype}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path} (and common variants).")
        print("Using a dummy image for demonstration purposes as a fallback.")
        # Create a dummy image (e.g., 256x256 RGB)
        img_np = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        original_image_for_plot = img_np.copy()
        image_filename = "Dummy Image"
        print(f"Dummy image shape: {img_np.shape}, dtype: {img_np.dtype}")
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        print("Exiting.")
        exit()

    # --- Perform prediction ---
    if img_np is not None:
        print("Performing prediction...")
        prediction_map = wrapper.predict(img_np) # img_np will be processed by predict method
        print("Prediction finished.")

        # --- Display or save the prediction map ---
        print(f"Prediction map shape: {prediction_map.shape}, dtype: {prediction_map.dtype}")
        print(f"Unique predicted classes: {np.unique(prediction_map)}")

        try:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            # Display original image 
            if original_image_for_plot.ndim == 3 and original_image_for_plot.shape[2] >= 3:
                plt.imshow(original_image_for_plot[:,:,:3]) # Show RGB
            elif original_image_for_plot.ndim == 2: # Grayscale
                plt.imshow(original_image_for_plot, cmap='gray')
            else: # Fallback for other complex cases (e.g. multi-channel non-RGB)
                 plt.imshow(original_image_for_plot[:,:,0], cmap='gray') # Show first channel
            plt.title(f"Original Image ({os.path.basename(image_path)})")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(prediction_map, cmap='viridis') 
            plt.title("Prediction Map (Predicted Classes)")
            plt.colorbar(label="Predicted Class")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            
            # Example: To save the figure automatically:
            # output_viz_path = "prediction_visualization.png"
            # plt.savefig(output_viz_path)
            # print(f"Prediction visualization saved to {output_viz_path}")

            # Example: To save the raw prediction map as a NumPy array:
            # output_map_path = "prediction_map_raw.npy"
            # np.save(output_map_path, prediction_map)
            # print(f"Raw prediction map saved to {output_map_path}")

        except ImportError:
            print("Matplotlib not installed (or other display error). Skipping visualization.")
            print("You can save the prediction map using: np.save('prediction_map.npy', prediction_map)")
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        print("Skipping prediction as image could not be loaded.")

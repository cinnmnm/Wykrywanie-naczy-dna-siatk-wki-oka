
from Util.config import Config
import os

class DatasetSupplier:
    @staticmethod
    def get_dataset(config_path='config.yaml'):
        """
        Returns a dataset based on the provided dataset name.
        """
        config = Config.load(config_path)
        image_dir = config["image_dir"]
        images_path = os.path.join(image_dir, "pictures")
        manual_path = os.path.join(image_dir, "manual")
        mask_path = os.path.join(image_dir, "mask")

        dataset = []

        for img_filename in os.listdir(images_path):
            base_name, img_ext = os.path.splitext(img_filename)
            
            potential_manual_exts = ['.gif', '.tif', '.png'] 
            actual_manual_file = None
            for ext in potential_manual_exts:
                potential_manual_name = base_name + ext
                if os.path.exists(os.path.join(manual_path, potential_manual_name)):
                    actual_manual_file = os.path.join(manual_path, potential_manual_name)
                    break

            potential_mask_exts = ['.gif', '.png', '.tif'] 
            actual_mask_file = None
            for ext in potential_mask_exts:
                potential_mask_name = base_name + '_mask' + ext 
                if os.path.exists(os.path.join(mask_path, potential_mask_name)):
                    actual_mask_file = os.path.join(mask_path, potential_mask_name)
                    break

            img_full_path = os.path.join(images_path, img_filename)

            if actual_manual_file and actual_mask_file and os.path.exists(img_full_path):
                dataset.append((base_name, img_full_path, actual_manual_file, actual_mask_file))
            else:
                print(f"Warning: Could not find matching files for base {base_name}")
                if not os.path.exists(img_full_path):
                    print(f"  Image file missing: {img_full_path}")
                if not actual_manual_file:
                    print(f"  Manual file missing for base: {base_name} in {manual_path}")
                if not actual_mask_file:
                    print(f"  Mask file missing for base: {base_name} in {mask_path}")

        return dataset
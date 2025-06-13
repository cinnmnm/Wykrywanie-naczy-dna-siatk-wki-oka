from Data.DatasetSupplier import DatasetSupplier
from Data.PatchFeatureExtractor import PatchFeatureExtractor
from Util.ImageLoader import ImageLoader
import numpy as np
from Data.Preprocessing import ImagePreprocessing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class MLInferenceDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        dataset = DatasetSupplier.get_dataset()
        tuple =  next((item for item in dataset if item[1].lower() == image_path.lower()), None)
        if tuple is None:
            raise Exception(f"wrong image path: {image_path}")
        _, image_path, manual_path, mask_path = tuple
        images_list = ImageLoader.load_images([image_path], BGRtoRGB=True)
        manual_list = ImageLoader.load_images([manual_path])
        masks_list = ImageLoader.load_images([mask_path])

        image = np.stack(images_list[0])
        manual = np.stack(manual_list[0])
        mask = np.stack(masks_list[0])

        image_processed = self.preprocessing(image)

        image_resized = ImagePreprocessing.resize_and_normalize(image_processed)
        manual_resized = ImagePreprocessing.resize_and_normalize(manual)
        mask_resized = ImagePreprocessing.resize_and_normalize(mask)

        pfe = PatchFeatureExtractor()
        patches, lables, coords = pfe.extract_patches([image_resized], [manual_resized], [mask_resized], patch_size=5, all_patches=True)

        features = np.stack([pfe.extract_features(patch) for patch in patches])
        
        ss = StandardScaler()
        scaled_features = ss.fit_transform(features)

        self.X = scaled_features
        self.Y = lables
        self.coords = coords

    def preprocessing(self, image):
        clahe = ImagePreprocessing.apply_clahe(image)
        median_filtered = ImagePreprocessing.median_filter(clahe)
        normalized = ImagePreprocessing.normalize(median_filtered)
        return normalized

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.coords[idx]
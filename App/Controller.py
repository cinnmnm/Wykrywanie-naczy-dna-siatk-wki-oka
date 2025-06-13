from Data.DatasetSupplier import DatasetSupplier
from Data.FilterSegmentation import FilterSegmentation
import numpy as np
import cv2
import pickle
import os
from Data.Preprocessing import ImagePreprocessing
from MLInferenceDataset import MLInferenceDataset
from DLUnet.Core.dataset import VesselSegmentationDataset
from DLUnet.Core.inference import VesselSegmentationInference
from torch.utils.data import DataLoader
from Util.config import Config

class Controller:
    def run_filter(self, image):
        return FilterSegmentation.run(image)

    def run_ml(self, img_path):
        img_path = img_path.replace("\\", "/", 1)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'DFModel', 'SavedModels', 'random_forest_model.pkl')
        model_path = os.path.abspath(model_path)

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception:
            return

        try:
            dataset = MLInferenceDataset(img_path)
        except Exception:
            return

        predictions = []
        pred_image = np.zeros((512, 512), dtype=np.uint8)

        for i in range(len(dataset)):
            try:
                X, Y, coords = dataset[i]
                pred = model.predict([X])
                predictions.append(pred)
                pred_image[coords[0]][coords[1]] = pred * 255
            except Exception:
                pass

        return pred_image

    def run_dl(self, img_path):
        config = Config.load('DLUnet/config_unet.yaml')
        dataset = DatasetSupplier.get_dataset('DLUnet/config_unet.yaml')
        
        basename = os.path.basename(img_path).split('.')[0]
        print(basename)

        test_samples = [item for item in dataset if item[0] in [basename]]
        data_tuples = [(img_path, mask_path, label_path) for (base, img_path, label_path, mask_path) in test_samples]
        test_dataset = VesselSegmentationDataset(
            data_tuples,
            image_transform=None,  # Use default transform (to tensor, normalization)
            target_size=tuple(config['target_size']),
            global_contrast_normalization=config.get('preprocessing', {}).get('global_contrast_normalization', False)
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        inference_engine = VesselSegmentationInference(
            model_path='DLUnet/SavedModels/demo_model_sota.pth',
            config_path='DLUnet/config_unet.yaml',
            device=config.get('device', None)
        )

        batch = next(iter(test_loader))
        img = batch['image'][0]  # [3, H, W]
        mask = batch['mask'][0][0].cpu().numpy()

        # Run inference using Core inference (single image)
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        img_np_uint8 = (img_np * 255).astype(np.uint8)
        pred_bin, pred_prob = inference_engine.predict_single(
            image=img_np_uint8,
            mask=None,
            threshold=config['inference']['threshold'],
            return_probability=True
        )
        pred_bin_masked = np.where(mask > 0.5, pred_bin, 0).astype(np.uint8)
        pred_map = pred_bin_masked * 255

        return pred_map
    
    def dummy(self, image):
        image = ImagePreprocessing.resize_and_normalize(image)
        image = (image > 0).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

"""
Decision Forest Pipeline for Retinal Vessel Segmentation

This module provides the main pipeline class that combines training and inference
capabilities for decision forest-based vessel segmentation.
"""

import os
from typing import List, Tuple, Optional, Dict
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .Core.dataset import PatchDataset, ImageDataset
from .Core.feature_extractor import PatchFeatureExtractor
from .Core.model import DecisionForestClassifier
from .Core.trainer import DecisionForestTrainer
from .Core.inference import DecisionForestInference
from Util.config import Config

class DecisionForestPipeline:
    """
    Main pipeline class for retinal vessel segmentation using Decision Forest.
    Provides a unified interface for training, evaluation, and inference.
    """
    def __init__(self, config_path: str = "DFModel/config_dfmodel.yaml"):
        self.config_path = config_path
        self.config = Config.load(config_path)
        self.dataset = None  # Will be set by prepare_data
        self.trainer = DecisionForestTrainer(config_path)
        self.inference_engine = None
        print(f"[DecisionForestPipeline] Initialized with config: {config_path}")

    def prepare_data(self, image_dataset: ImageDataset, patch_size=None, samples_per_image=None, positive_ratio=0.5):
        """
        Prepare PatchDataset from an ImageDataset.
        """
        print(f"[DecisionForestPipeline] Preparing data...")
        patch_size = patch_size or self.config.get('patch', {}).get('size', 27)
        samples_per_image = samples_per_image or self.config.get('patch', {}).get('samples_per_image', 1000)
        
        # Create feature extractor with config
        feature_extractor = PatchFeatureExtractor(self.config.get('features', {}))
        
        # Create PatchDataset using proven extraction logic
        self.dataset = PatchDataset(
            image_dataset, 
            patch_size=patch_size, 
            feature_extractor=feature_extractor, 
            samples_per_image=samples_per_image, 
            positive_ratio=positive_ratio
        )
        
        print(f"[DecisionForestPipeline] Data preparation completed. Features shape: {self.dataset.features.shape if self.dataset.features is not None else None}")

    def initialize_model(self):
        print(f"[DecisionForestPipeline] Initializing model...")
        return self.trainer.initialize_model()

    def train(self, hyperparameter_tuning: bool = False):
        print(f"[DecisionForestPipeline] Starting training...")
        if self.dataset is None or self.dataset.features is None:
            raise ValueError("Must call prepare_data() first")
        
        train_metrics = self.trainer.train(
            dataset=self.dataset,
            hyperparameter_tuning=hyperparameter_tuning
        )
        self.trainer.save_model()
        print(f"[DecisionForestPipeline] Training completed")
        return train_metrics

    def evaluate(self):
        print(f"[DecisionForestPipeline] Evaluating model...")
        if self.dataset is None or self.dataset.features is None:
            raise ValueError("Must call prepare_data() first")
        return self.trainer.evaluate(self.dataset)

    def cross_validate(self, cv_folds: int = 5):
        print(f"[DecisionForestPipeline] Performing cross-validation...")
        return self.trainer.cross_validate(self.dataset, cv_folds)

    def load_weights(self, path: str):
        print(f"[DecisionForestPipeline] Loading model from: {path}")
        self.trainer.load_model(path)

    def plot_training_results(self):
        print(f"[DecisionForestPipeline] Plotting results...")
        self.trainer.plot_feature_importance()
        summary = self.trainer.get_training_summary()
        print(f"[DecisionForestPipeline] Training Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value:.4f}" if isinstance(sub_value, float) else f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

    def initialize_inference(self, model_path: Optional[str] = None):
        model_path = model_path or self.config.get('model_save_path', 'DFModel/SavedModels/dfmodel_best.pkl')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.inference_engine = DecisionForestInference(model_path, self.config_path)
        print(f"[DecisionForestPipeline] Inference engine initialized")

    def predict(self, image_mask_pairs: List[Tuple[str, Optional[str]]], output_dir: Optional[str] = None, postprocess: bool = True) -> List[np.ndarray]:
        if self.inference_engine is None:
            self.initialize_inference()
        print(f"[DecisionForestPipeline] Running inference on {len(image_mask_pairs)} images...")
        image_paths = [pair[0] for pair in image_mask_pairs]
        mask_paths = [pair[1] if len(pair) > 1 and pair[1] is not None else None for pair in image_mask_pairs]
        
        # Filter out None mask paths for the inference engine
        filtered_mask_paths = [path for path in mask_paths if path is not None] if any(path is not None for path in mask_paths) else None
        
        predictions = self.inference_engine.predict_from_paths(image_paths, filtered_mask_paths)
        if postprocess:
            print(f"[DecisionForestPipeline] Applying postprocessing...")
            predictions = [self.inference_engine.postprocess_predictions(pred) for pred in predictions]
        if output_dir:
            image_names = [os.path.basename(path) for path in image_paths]
            self.inference_engine.save_predictions(predictions, output_dir, image_names)
        return predictions

    def predict_single(self, image_path: str, mask_path: Optional[str] = None) -> np.ndarray:
        if self.inference_engine is None:
            self.initialize_inference()
        mask_paths = [mask_path] if mask_path else None
        return self.inference_engine.predict_from_paths([image_path], mask_paths)[0]

    def get_model_info(self) -> Dict:
        if self.trainer.model is not None:
            return self.trainer.model.get_model_info()
        elif self.inference_engine is not None:
            return self.inference_engine.get_model_info()
        else:
            return {"status": "not_initialized"}

    def export_features(self, output_path: str):
        if self.dataset is None or self.dataset.features is None:
            raise ValueError("Features not available. Call prepare_data() first.")
        
        # Get feature names
        if len(self.dataset.patches) > 0:
            feature_names = self.dataset.feature_extractor.get_feature_names(self.dataset.patches[0].shape)
        else:
            feature_names = [f"feature_{i}" for i in range(self.dataset.features.shape[1])]
        
        # Save to CSV
        import csv
        print(f"[DecisionForestPipeline] Saving features to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = feature_names + ['label']
            writer.writerow(header)
            for feature_row, label in zip(self.dataset.features, self.dataset.patch_labels):
                row = feature_row.tolist() + [label]
                writer.writerow(row)
        
        print(f"[DecisionForestPipeline] Features exported to: {output_path}")

    def get_feature_statistics(self) -> Dict:
        if self.dataset is None or self.dataset.features is None:
            return {}
        
        return {
            'n_samples': len(self.dataset.features),
            'n_features': self.dataset.features.shape[1] if len(self.dataset.features) > 0 else 0,
            'n_positive': np.sum(self.dataset.patch_labels == 1),
            'n_negative': np.sum(self.dataset.patch_labels == 0),
            'feature_means': np.mean(self.dataset.features, axis=0) if len(self.dataset.features) > 0 else [],
            'feature_stds': np.std(self.dataset.features, axis=0) if len(self.dataset.features) > 0 else []
        }

# Factory function for easy instantiation
def create_decision_forest_pipeline(config_path: str = "DFModel/config_dfmodel.yaml") -> DecisionForestPipeline:
    return DecisionForestPipeline(config_path)

# Convenience function for quick inference
def run_decision_forest_classification(image_mask_pairs: List[Tuple[str, Optional[str]]], model_path: str, config_path: str = "DFModel/config_dfmodel.yaml") -> List[np.ndarray]:
    pipeline = DecisionForestPipeline(config_path)
    pipeline.initialize_inference(model_path)
    return pipeline.predict(image_mask_pairs)

"""
Decision Forest Trainer

This module provides training capabilities for decision forest models
using patch-based features for retinal vessel classification.
"""

import os
import time
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from Util.config import Config
from .model import DecisionForestClassifier
from .dataset import PatchDataset


class DecisionForestTrainer:
    """
    Trainer class for Decision Forest vessel classification.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = Config.load(config_path)
        
        # Initialize paths
        self.model_save_path = self.config.get('model_save_path', 'DFModel/SavedModels/dfmodel_best.pkl')
        self.results_dir = self.config.get('results_dir', 'DFModel/Results/')
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.model = None
        self.training_time = 0
        self.feature_names = []
        self.best_params = {}
        
        print(f"[DecisionForestTrainer] Initialized")
        print(f"  Config: {os.path.basename(config_path)}")
        print(f"  Model save path: {self.model_save_path}")
        print(f"  Results directory: {self.results_dir}")
    
    def initialize_model(self):
        """Initialize the Decision Forest model"""
        self.model = DecisionForestClassifier(self.config)
        print(f"[DecisionForestTrainer] Model initialized successfully")
        return self.model
        
    def train(self, dataset: PatchDataset, hyperparameter_tuning: bool = False) -> Dict[str, float]:
        """
        Train the Decision Forest model.
        
        Args:
            dataset: Training dataset (PatchDataset)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Training metrics
        """
        if self.model is None:
            self.initialize_model()
        
        print(f"[DecisionForestTrainer] Starting training...")
        start_time = time.time()
        
        # Get training data from PatchDataset
        X_train, y_train = dataset.get_features_and_labels()
        self.feature_names = dataset.feature_extractor.get_feature_names(dataset.patches[0].shape) if dataset.patches is not None and len(dataset.patches) > 0 else []
        
        print(f"[DecisionForestTrainer] Training data: {X_train.shape}")
        print(f"[DecisionForestTrainer] Positive samples: {np.sum(y_train == 1)}, Negative samples: {np.sum(y_train == 0)}")
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            print(f"[DecisionForestTrainer] Performing hyperparameter tuning...")
            self._hyperparameter_tuning(X_train, y_train)
        
        # Train the model
        self.model.fit(X_train, y_train, self.feature_names)
        
        # Calculate training time
        self.training_time = time.time() - start_time
        print(f"[DecisionForestTrainer] Training completed in {self.training_time:.2f} seconds")
        
        # Evaluate on training data
        train_metrics = self.model.evaluate(X_train, y_train)
        print(f"[DecisionForestTrainer] Training metrics: {train_metrics}")
        
        return train_metrics
    
    def evaluate(self, test_dataset: PatchDataset) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Test metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print(f"[DecisionForestTrainer] Evaluating model...")
        
        # Get test data from PatchDataset
        X_test, y_test = test_dataset.get_features_and_labels()
        
        # Evaluate
        test_metrics = self.model.evaluate(X_test, y_test)
        print(f"[DecisionForestTrainer] Test metrics: {test_metrics}")
        
        return test_metrics
    
    def cross_validate(self, dataset: PatchDataset, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            dataset: Dataset for cross-validation
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        if self.model is None:
            self.initialize_model()
        
        print(f"[DecisionForestTrainer] Performing {cv_folds}-fold cross-validation...")
        
        # Get data from PatchDataset
        X, y = dataset.get_features_and_labels()
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model.model, X, y, cv=cv_folds, scoring='f1')
        
        results = {
            'cv_f1_mean': np.mean(cv_scores),
            'cv_f1_std': np.std(cv_scores),
            'cv_f1_scores': cv_scores.tolist()
        }
        
        print(f"[DecisionForestTrainer] Cross-validation results: {results}")
        return results
    
    def _hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Labels
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        print(f"[DecisionForestTrainer] Best parameters: {self.best_params}")
        
        # Update model with best parameters
        self.model.model = grid_search.best_estimator_
    
    def save_model(self, save_path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            save_path: Optional custom save path
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        save_path = save_path or self.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        self.model.save_model(save_path)
        print(f"[DecisionForestTrainer] Model saved to: {save_path}")
    
    def load_model(self, load_path: Optional[str] = None):
        """
        Load a trained model.
        
        Args:
            load_path: Optional custom load path
        """
        load_path = load_path or self.model_save_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        if self.model is None:
            self.initialize_model()
        
        self.model.load_model(load_path)
        print(f"[DecisionForestTrainer] Model loaded from: {load_path}")
    
    def plot_feature_importance(self, top_n: int = 20):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        feature_importance = self.model.get_feature_importance()
        if not feature_importance:
            print("No feature importance available")
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Plot
        features, importances = zip(*top_features)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"[DecisionForestTrainer] Feature importance plot saved to: {plot_path}")
    
    def get_training_summary(self) -> Dict:
        """
        Get a summary of the training process.
        
        Returns:
            Training summary dictionary
        """
        return {
            'model_type': 'RandomForest',
            'training_time': self.training_time,
            'feature_count': len(self.feature_names),
            'best_params': self.best_params,
            'config_path': self.config_path
        }

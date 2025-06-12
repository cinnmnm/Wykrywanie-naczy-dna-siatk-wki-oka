"""
Decision Forest Model for Retinal Vessel Classification

This module implements a Random Forest/Decision Forest classifier
for vessel classification based on extracted patch features.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Util.config import Config


class DecisionForestClassifier:
    """
    Decision Forest classifier for retinal vessel classification.
    
    This class wraps scikit-learn's RandomForestClassifier with
    vessel-specific optimizations and evaluation metrics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Decision Forest classifier.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self.config = config or {}
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        # Initialize Random Forest with configuration
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Random Forest model with configuration"""
        model_config = self.config.get('model', {})
        
        # Default Random Forest parameters optimized for vessel classification
        rf_params = {
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', None),
            'min_samples_split': model_config.get('min_samples_split', 2),
            'min_samples_leaf': model_config.get('min_samples_leaf', 1),
            'max_features': model_config.get('max_features', 'sqrt'),
            'bootstrap': model_config.get('bootstrap', True),
            'n_jobs': model_config.get('n_jobs', -1),  # Use all cores
            'random_state': model_config.get('random_state', 42),
            'class_weight': model_config.get('class_weight', 'balanced'),  # Handle imbalanced data
            'verbose': model_config.get('verbose', 0)
        }
        
        self.model = RandomForestClassifier(**rf_params)
        
        print(f"[DecisionForestClassifier] Initialized Random Forest:")
        print(f"  n_estimators: {rf_params['n_estimators']}")
        print(f"  max_depth: {rf_params['max_depth']}")
        print(f"  class_weight: {rf_params['class_weight']}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        Train the Decision Forest classifier.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            feature_names: Optional list of feature names
        """
        if self.model is None:
            self._initialize_model()
        
        print(f"[DecisionForestClassifier] Training on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names for later analysis
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        print(f"[DecisionForestClassifier] Training completed")
        
        # Print feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self._print_feature_importance()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred)),
            'auc': float(roc_auc_score(y, y_proba[:, 1])) if y_proba.shape[1] > 1 else 0.0
        }
        
        print("\n[DecisionForestClassifier] Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        return importance_dict
    
    def _print_feature_importance(self, top_k: int = 10) -> None:
        """Print top k most important features"""
        importance_dict = self.get_feature_importance()
        if importance_dict is None:
            return
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n[DecisionForestClassifier] Top {top_k} Feature Importances:")
        for i, (feature, importance) in enumerate(sorted_features[:top_k]):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to file.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        print(f"[DecisionForestClassifier] Model saved to: {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', {})
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"[DecisionForestClassifier] Model loaded from: {path}")
    
    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'is_trained': self.is_trained,
            'feature_names_available': self.feature_names is not None
        }
        
        return info
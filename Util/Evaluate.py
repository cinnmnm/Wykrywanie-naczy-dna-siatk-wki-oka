import numpy as np
from sklearn.metrics import confusion_matrix
import yaml
import os

class Evaluate:
    @staticmethod
    def confusion(y_true, y_pred, mask=None):
        """
        Return confusion matrix (TN, FP, FN, TP) for binary classification.
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Ensure y_true_flat and y_pred_flat are binary (0 or 1)
        y_true_flat = np.where(y_true_flat > 1, 1, y_true_flat)
        y_pred_flat = np.where(y_pred_flat > 1, 1, y_pred_flat)

        if mask is not None:
            mask_flat = mask.flatten() != 0
            if mask_flat.size != y_true_flat.size:
                # This basic check assumes mask, if provided, should correspond to y_true/y_pred.
                # More sophisticated error handling or assumptions about broadcasting might be needed
                # depending on expected use cases for differing shapes.
                raise ValueError(f"Mask shape incompatible with y_true/y_pred shape after flattening. y_true_flat.size: {y_true_flat.size}, mask_flat.size: {mask_flat.size}")
            y_true_masked = y_true_flat[mask_flat]
            y_pred_masked = y_pred_flat[mask_flat]
        else:
            y_true_masked = y_true_flat
            y_pred_masked = y_pred_flat
        
        if y_true_masked.size == 0:
            # If no elements to compare (e.g. empty inputs or mask filters out everything)
            return {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}

        # Ensure labels=[0,1] to get a 2x2 matrix even if some classes are not present
        # in y_true_masked or y_pred_masked.
        cm_values = confusion_matrix(y_true_masked, y_pred_masked, labels=[0,1]).ravel()
        tn, fp, fn, tp = cm_values

        return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

    @staticmethod
    def accuracy(y_true, y_pred, mask=None):
        """
        Computes accuracy using the confusion matrix.
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        tp = cm['TP']
        tn = cm['TN']
        fp = cm['FP']
        fn = cm['FN']
        
        total = tp + tn + fp + fn
        if total == 0:
            return 0.0 
        
        return (tp + tn) / total

    @staticmethod
    def sensitivity(y_true, y_pred, mask=None):
        """
        Computes sensitivity (True Positive Rate or Recall).
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        tp = cm['TP']
        fn = cm['FN']
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def specificity(y_true, y_pred, mask=None):
        """
        Computes specificity (True Negative Rate).
        If mask is provided, only evaluates pixels where mask != 0.
        Otherwise, all pixels are evaluated.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        tn = cm['TN']
        fp = cm['FP']
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def print_confusion_matrix(y_true, y_pred, mask=None):
        """
        Calls the confusion method and prints the confusion matrix in a readable format.
        """
        cm = Evaluate.confusion(y_true, y_pred, mask)
        print("Confusion Matrix:")
        print(f"TN: {cm['TN']}  FP: {cm['FP']}")
        print(f"FN: {cm['FN']}  TP: {cm['TP']}")


import numpy as np
from sklearn.metrics import confusion_matrix

class Evaluator:
    @staticmethod
    def confusion(y_true, y_pred, mask):
        """
        Return confusion matrix (TN, FP, FN, TP) for binary classification.
        Only evaluates pixels where mask != 0.
        """
        mask_flat = mask.flatten() != 0
        y_true_masked = y_true.flatten()[mask_flat]
        y_pred_masked = y_pred.flatten()[mask_flat]
        tn, fp, fn, tp = confusion_matrix(y_true_masked, y_pred_masked, labels=[0,1]).ravel()
        return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

    @staticmethod
    def accuracy(y_true, y_pred, mask):
        """
        Computes accuracy.
        Only evaluates pixels where mask != 0.
        """
        mask_flat = mask.flatten() != 0
        y_true_masked = y_true.flatten()[mask_flat]
        y_pred_masked = y_pred.flatten()[mask_flat]
        correct = np.sum(y_true_masked == y_pred_masked)
        total = y_true_masked.size
        return correct / total if total > 0 else 0.0

    @staticmethod
    def sensitivity(y_true, y_pred, mask):
        """
        Computes sensitivity.
        Only evaluates pixels where mask != 0.
        """
        cm = Evaluator.confusion(y_true, y_pred, mask)
        tp = cm['TP']
        fn = cm['FN']
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def specificity(y_true, y_pred, mask):
        """
        Computes specificity.
        Only evaluates pixels where mask != 0.
        """
        cm = Evaluator.confusion(y_true, y_pred, mask)
        tn = cm['TN']
        fp = cm['FP']
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
"""
Integration with Existing Evaluation Pipeline
============================================

This shows how to integrate FastInferenceEngine with your existing
Evaluate class and evaluation metrics.
"""

from Util.Evaluate import Evaluate
from Data.FastInferenceEngine import FastInferenceEngine
import torch
import numpy as np

class FastEvaluate(Evaluate):
    """
    Enhanced Evaluate class that uses FastInferenceEngine for evaluation.
    Drop-in replacement for your existing Evaluate class.
    """
    
    def __init__(self, model, gpu_dataset, device='cuda'):
        super().__init__()
        self.model = model
        self.gpu_dataset = gpu_dataset
        self.device = device
        
        # Initialize fast inference engine
        self.fast_engine = FastInferenceEngine(
            model=model,
            device=device,
            patch_size=27,  # Use your config value
            batch_size=2048
        )
    
    def evaluate_model_fast(self, image_indices=None):
        """
        Fast evaluation using vectorized inference.
        
        Args:
            image_indices: List of image indices to evaluate (None = all images)
        
        Returns:
            Dictionary with evaluation metrics
        """
        if image_indices is None:
            image_indices = list(range(len(self.gpu_dataset.images)))
        
        print(f"Fast evaluation on {len(image_indices)} images...")
        
        all_predictions = []
        all_ground_truth = []
        total_time = 0
        
        for img_idx in image_indices:
            print(f"Evaluating image {img_idx}...")
            
            # Fast inference
            start_time = time.time()
            pred_map = self.fast_engine.predict_full_image(
                self.gpu_dataset, img_idx, overlap_strategy='center'
            )
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Get ground truth
            gt_labels = self.gpu_dataset.labels[img_idx]
            if gt_labels.ndim == 3:
                gt_labels = gt_labels[0]
            
            # Get valid mask region
            mask = self.gpu_dataset.masks[img_idx]
            if mask.ndim == 3:
                mask = mask[0]
            
            # Only evaluate in valid regions
            valid_region = mask > 0.5
            if valid_region.sum() > 0:
                pred_valid = pred_map[valid_region].cpu().numpy()
                gt_valid = (gt_labels[valid_region] > 0.5).cpu().numpy().astype(int)
                
                all_predictions.extend(pred_valid)
                all_ground_truth.extend(gt_valid)
            
            print(f"  Time: {inference_time:.2f}s")
        
        # Calculate metrics using your existing methods
        predictions = np.array(all_predictions)
        ground_truth = np.array(all_ground_truth)
        
        # Use your existing metric calculations
        accuracy = (predictions == ground_truth).mean()
        
        # Calculate additional metrics if you have them
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(ground_truth, predictions, average='binary')
        recall = recall_score(ground_truth, predictions, average='binary')
        f1 = f1_score(ground_truth, predictions, average='binary')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_time': total_time,
            'avg_time_per_image': total_time / len(image_indices),
            'total_pixels_evaluated': len(predictions),
            'images_evaluated': len(image_indices)
        }
        
        print(f"\nFast Evaluation Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per image: {total_time/len(image_indices):.2f}s")
        
        return results

# Usage example:
def integrate_with_existing_evaluation():
    """Show how to use FastEvaluate with your existing code."""
    
    # Your existing setup
    from DLPatch.DLModel import DLModel
    from Data.GPUMappedDataset import GPUMappedDataset
    from Data.DatasetSupplier import DatasetSupplier
    
    # Load your model
    model = DLModel()
    model.load_state_dict(torch.load("DLPatch/SavedModels/model_final.pth"))
    model.eval()
    
    # Setup GPU dataset
    dataset_tuples = DatasetSupplier.get_dataset()
    gpu_dataset = GPUMappedDataset(dataset_tuples, device='cuda')
    
    # Use fast evaluation (drop-in replacement)
    fast_evaluator = FastEvaluate(model, gpu_dataset)
    
    # Evaluate on all images (or subset)
    results = fast_evaluator.evaluate_model_fast()
    
    # Results are compatible with your existing analysis
    return results

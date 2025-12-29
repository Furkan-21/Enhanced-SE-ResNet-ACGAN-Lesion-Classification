# src/evaluate.py
"""
Evaluation and prediction functions for skin lesion classification models.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from PIL import Image
from typing import Any, List, Optional, Tuple, Dict

def evaluate_model(model: torch.nn.Module, test_loader: Any, device: torch.device, categories: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset with comprehensive metrics for imbalanced classification.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on (CPU or GPU)
        categories: Optional list of category names
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate overall metrics with different averaging methods
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
        'precision_macro': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'precision_weighted': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        'recall_macro': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall_weighted': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
    }
    
    # Calculate per-class metrics
    unique_classes = sorted(set(all_targets).union(set(all_preds)))
    for cls in unique_classes:
        cls_targets = [1 if t == cls else 0 for t in all_targets]
        cls_preds = [1 if p == cls else 0 for p in all_preds]
        metrics[f'f1_class_{cls}'] = f1_score(cls_targets, cls_preds, zero_division=0)
        metrics[f'precision_class_{cls}'] = precision_score(cls_targets, cls_preds, zero_division=0)
        metrics[f'recall_class_{cls}'] = recall_score(cls_targets, cls_preds, zero_division=0)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Generate classification report
    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=categories if categories else [str(i) for i in unique_classes],
        zero_division=0,
        output_dict=True
    )
    
    # Print summary metrics
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    # Print per-class metrics
    print("\nPer-Class F1 Scores:")
    for cls in unique_classes:
        class_name = categories[cls] if categories and cls < len(categories) else str(cls)
        print(f"  Class {class_name}: {metrics[f'f1_class_{cls}']:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Store additional information in metrics dictionary
    metrics['confusion_matrix'] = cm.tolist()
    metrics['classification_report_dict'] = report # Store the dict version
    
    # Generate string version of classification report for direct printing if needed
    class_report_str = classification_report(
        all_targets, 
        all_preds, 
        target_names=categories if categories else [str(i) for i in unique_classes],
        zero_division=0,
        output_dict=False # Get string output
    )
    
    return metrics, cm, class_report_str

def predict_single_image(model: torch.nn.Module, image_path: str, transform: Any, device: torch.device, categories: Optional[List[str]] = None) -> Tuple[str, float]:
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    predicted_class = categories[predicted.item()] if categories else str(predicted.item())
    return predicted_class, confidence.item()

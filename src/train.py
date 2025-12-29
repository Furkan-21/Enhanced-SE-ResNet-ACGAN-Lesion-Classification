# src/train.py
"""
Training and validation logic for skin lesion classification models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def compute_class_weights(labels: list) -> torch.Tensor:
    unique_classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique_classes)
    weights = n_samples / (n_classes * counts)
    weights = weights * n_classes / weights.sum()
    return torch.FloatTensor(weights)

def compute_metrics(targets: list, predictions: list) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'f1_macro': f1_score(targets, predictions, average='macro'),
        'f1_weighted': f1_score(targets, predictions, average='weighted'),
        'precision_macro': precision_score(targets, predictions, average='macro'),
        'recall_macro': recall_score(targets, predictions, average='macro'),
    }
    unique_classes = np.unique(targets + predictions)
    for cls in unique_classes:
        cls_targets = [1 if t == cls else 0 for t in targets]
        cls_preds = [1 if p == cls else 0 for p in predictions]
        metrics[f'f1_class_{cls}'] = f1_score(cls_targets, cls_preds)
        metrics[f'precision_class_{cls}'] = precision_score(cls_targets, cls_preds)
        metrics[f'recall_class_{cls}'] = recall_score(cls_targets, cls_preds)
    return metrics

def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: torch.device,
    num_epochs: int = 50,
    patience: int = 10,
    weight_classes: bool = True,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, list]]:
    print("train_model: Entered function.")

    if weight_classes:
        print("train_model: weight_classes is True. Collecting all training labels for weight calculation...")
        all_train_labels = []
        # Wrap the loop with tqdm for progress visibility
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Collecting training labels")):
            all_train_labels.extend(batch['label'].cpu().numpy())
            # Optional: print progress every N batches if tqdm isn't enough
            # if (batch_idx + 1) % 50 == 0:
            #     print(f"train_model: Processed {batch_idx + 1} batches for label collection.")

        print(f"train_model: Finished collecting {len(all_train_labels)} training labels.")

        if not all_train_labels:
            print("train_model: Warning - no training labels collected. Class weighting will likely fail or be incorrect.")
            # Fallback to no weights if no labels were collected, or handle as an error
            # For now, let it proceed; compute_class_weights might raise an error or return problematic weights.

        class_weights = compute_class_weights(all_train_labels).to(device)
        print(f"train_model: Using class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("train_model: weight_classes is False. Using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001) # Consider making lr a parameter
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    print("train_model: Starting training epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_metrics = compute_metrics(train_targets, train_preds)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation'):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_metrics = compute_metrics(val_targets, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_f1'].append(val_metrics['f1_macro'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_metrics': val_metrics,
                }, save_path)
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train F1: {train_metrics['f1_macro']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")
        for cls in sorted([k.split('_')[-1] for k in val_metrics.keys() if k.startswith('f1_class_')]):
            print(f"  Class {cls} F1: {val_metrics[f'f1_class_{cls}']:.4f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs without improvement")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)

    return model, history

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_loader, classes, config, run_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    report_dir = os.path.join(config['training']['reports_dir'], run_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate classification report
    report_str = classification_report(all_labels, all_preds, target_names=classes)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    with open(os.path.join(report_dir, f"classification_report_{run_name}.txt"), "w") as f:
        f.write(report_str)
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {run_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(report_dir, f'confusion_matrix_{run_name}.png')
    plt.savefig(cm_path)
    plt.close()
    
    return {
        "run_name": run_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm_path": cm_path
    }

# imports
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torchmetrics
from typing import Tuple

logger = logging.getLogger(__name__)

def evaluate(model: nn.Module, 
            test_metric: torchmetrics.Metric,
            is_binary: bool,
            device: torch.device, 
            dataloader: DataLoader, 
            criterion: nn.Module
            ) -> Tuple[
                float, float, list[int], matplotlib.figure.Figure
                ]:
    '''
    Evaluates a trained PyTorch model on a test dataset and visualizes the results.

    Parameters:
        model (nn.Module): 
            The trained PyTorch model to evaluate.
        test_metric (torchmetrics.Metric): 
            A metric object used to evaluate model performance on the test set.
        is_binary (bool): 
            Whether this is a binary classification task.
        device (torch.device): 
            The device to run the evaluation on (e.g., 'cpu' or 'cuda').
        dataloader (DataLoader): 
            DataLoader for the test dataset.
        criterion (nn.Module): 
            Loss function used to compute the test loss.

    Returns:
        tuple (float, float, list, fig):
            avg_test_loss (float): Loss on the test set
            test_metric_score (float): Metric score on the test set
            preds_list (list): List of model predictions.
            fig (matplotlib.figure.Figure): Confusion matrix plot object.
    '''
    model.eval()
    test_loss = 0.0
    preds_list = []
    labels_list = []

    test_metric.reset()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float() if is_binary else labels.long()

            outputs = model(inputs)
            if is_binary:
                outputs = outputs.view(-1)
                preds = torch.sigmoid(outputs) >= 0.5
            else:
                preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            test_metric.update(preds, labels)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(dataloader)
    test_metric_score = test_metric.compute()

    logger.info(
        f"Test Loss: {avg_test_loss:.4f}, Test metric: {test_metric_score:.4f}"
        )

    # Compute confusion matrix
    cm = confusion_matrix(labels_list, preds_list)

    # Get class names
    class_names = list(dataloader.dataset.label_map.keys())

    # Create and return the confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()

    return avg_test_loss, test_metric_score, preds_list, fig
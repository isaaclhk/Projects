# imports 
import logging
from copy import deepcopy
from collections import Counter
from tqdm import tqdm

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchmetrics

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, path, patience=5, delta=0.0):
        '''
        Parameters:
            path (str): Path to save the best model.
            patience (int): How long to wait after last time validation metric improved.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
        '''
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_max = -float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, metric_score: float, model: torch.nn.Module) -> None:
        '''
        Checks if validation metric has improved. Saves model if it has; otherwise, increments a counter.
        Triggers early stopping if no improvement for `patience` epochs.

        Parameters:
            metric_score (float): Current validation metric score.
            model (torch.nn.Module): Model to save if performance improves.
        '''
        score = metric_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: torch.nn.Module) -> None:
        '''Saves model when validation metric improves.'''
        logger.info("Validation metric improved. Exporting the model..")
        try:
            torch.save(model.state_dict(), self.path)
        except Exception as e:
            logger.warning("Failed to save model checkpoint.")

def train_step(
    model: torch.nn.Module, 
    train_metric: torchmetrics.Metric, 
    is_binary: bool,
    train_dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    optimizer: torch.optim.Optimizer, 
    loss_fn
    ) -> tuple:
    '''
    Performs a single training epoch for the given model.

    This function sets the model to training mode, iterates over the training dataloader,
    computes the loss, performs backpropagation, updates the model weights, and tracks
    a specified performance metric (e.g., recall).

    Parameters:
        model (torch.nn.Module): 
            The neural network model to be trained.
        train_metric (torchmetrics.Metric): 
            A metric object used to evaluate model performance during training.
        is_binary (bool):
            Whether this is a binary classification task.
        train_dataloader (torch.utils.data.DataLoader): 
            DataLoader providing batches of training data.
        device (torch.device): 
            The device (CPU or GPU) on which computations will be performed.
        optimizer (torch.optim.Optimizer): 
            Optimizer used to update model parameters.
        loss_fn (callable): 
            Loss function used to compute the error between predictions and true labels.

    Returns:
        batch_loss (float): 
            Average training loss over all batches in the epoch.
        train_metric_score (float): 
            Computed training metric score for the epoch.
    '''
    # Set model to training mode
    model.train()
    running_loss = 0.0
    train_metric.reset()  # Reset metric at the start of each epoch

    # Training loop
    for inputs, labels in tqdm(train_dataloader):
        # Move data to the target device and ensure correct data types
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.float() if is_binary else labels.long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        if is_binary:
            outputs = outputs.view(-1) # flatten logits

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        running_loss += loss.detach().cpu().item()

        # prediction and metric update
        if is_binary:
            preds = torch.sigmoid(outputs)
            preds = preds.unsqueeze(1)
        else:
            preds = torch.softmax(outputs, dim=1)
            
        train_metric.update(preds, labels)

    # compute average batch loss
    batch_loss = running_loss/len(train_dataloader)

    # Compute training recall for the epoch
    train_metric_score = train_metric.compute()

    return batch_loss, train_metric_score

def val_step(
    model: torch.nn.Module, 
    val_metric: torchmetrics.Metric, 
    is_binary: bool,
    val_dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    loss_fn
    ) -> tuple:
    '''
    Performs a single validation epoch for the given model.

    This function sets the model to evaluation mode, disables gradient computation,
    iterates over the validation dataloader, computes the loss, and evaluates a specified
    performance metric (e.g., recall) without updating model weights.

    Parameters:
        model (torch.nn.Module): 
            The neural network model being evaluated.
        val_metric (torchmetrics.Metric): 
            A metric object used to evaluate model performance on the validation set.
        is_binary (bool):
            Whether this is a binary classification task.
        val_dataloader (torch.utils.data.DataLoader): 
            DataLoader providing batches of validation data.
        device (torch.device): 
            The device (CPU or GPU) on which computations will be performed.
        loss_fn (callable): 
            Loss function used to compute the error between predictions and true labels.

    Returns:
        val_batch_loss (float): 
            Average validation loss over all batches in the epoch.
        val_metric_score (float): 
            Computed validation metric score for the epoch.
    '''
    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    val_metric.reset()  # Reset validation metric

    # Validation loop (no gradient computation)
    with torch.inference_mode():
        for inputs, labels in val_dataloader:
            # Move data to the target device and ensure correct data types
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float() if is_binary else labels.long()

            # Forward pass
            outputs = model(inputs)
            if is_binary:
                outputs = outputs.view(-1) # flatten logits

            # Compute validation loss
            loss = loss_fn(outputs, labels)
            running_loss += loss.detach().cpu().item()

            # prediction and metric update
            if is_binary:
                preds = torch.sigmoid(outputs)
                preds =  preds.insqueeze(1)
            else:
                preds = torch.softmax(outputs, dim=1)
                
            val_metric.update(preds, labels)

    # compute average batch loss
    batch_loss = running_loss/len(val_dataloader)

    # Compute validation metric for the epoch
    val_metric_score = val_metric.compute()

    return batch_loss, val_metric_score


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    '''
    Replaces the final classification layer of a given model 
    to match the specified number of output classes.

    Supports the EfficientNet architecture.
    Raises ValueError if the model architecture is not supported for classifier replacement.

    Parameters:
        model (nn.Module): The neural network model whose classifier is to be replaced.
        num_classes (int): The number of output classes for the new classifier.

    Returns:
        nn.Module: The model with the updated classifier layer.
    '''
    out_features = 1 if num_classes == 2 else num_classes
    logger.debug(
        f"Replacing classifier for model: {type(model).__name__}, target output classes: {out_features}"
    )

    # Replace EfficientNet classifier
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, out_features)
        )
        logger.debug(
            f"Replaced classifier in model with 'classifier' attribute. In: {in_features}, Out: {out_features}"
        )

    else:
        logger.error("Unsupported model architecture for classifier replacement.")
        raise ValueError("Unsupported model architecture for classifier replacement.")

    return model

def compute_class_weights(dataset: Dataset, num_classes: int) -> torch.Tensor:
    '''
    Computes class weights for imbalanced classification tasks using a PyTorch Dataset.

    The weights are calculated based on the inverse frequency of each class in the dataset,
    and normalized so that their sum equals the number of classes.

    Parameters:
        dataset (Dataset): A PyTorch Dataset that yields (input, label) pairs.
        num_classes (int): The total number of classes in the classification task.

    Returns:
        torch.Tensor: A 1D tensor of shape (num_classes,) containing the computed class weights.
    '''
    class_counts = Counter()

    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.view(-1).tolist()
        elif not isinstance(label, list):
            label = [label]
        class_counts.update(label)

    total_samples = sum(class_counts.values())

    weights = [0.0] * num_classes
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        weights[i] = total_samples / (count if count > 0 else 1)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * num_classes

    return weights



def set_seed(seed: int) -> None:
    '''
    Set the seed for random number generators in Python, NumPy, and PyTorch to ensure reproducibility.

    This function sets the seed for:
    - Python's built-in `random` module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators
    - Configures PyTorch to use deterministic algorithms for reproducibility

    Parameters:
        seed (int): The seed value to use for all random number generators.

    Returns:
        None
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

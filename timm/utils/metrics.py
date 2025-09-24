""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import numpy as np
# from sklearn.metrics import roc_auc_score
from typing import List, Optional

import torch
import torch.nn.functional as F

def roc_auc_score(y_true, y_score, average=None):
    device = y_score.device
    
    y_score = y_score.float()
    
    if y_true.dim() == 1:
        n_classes = y_score.shape[1]
        y_true_onehot = F.one_hot(y_true.long(), num_classes=n_classes).float()
    else:
        y_true_onehot = y_true.float()
    
    n_classes = y_true_onehot.shape[1]
    auc_scores = []
    valid_classes = []
    
    for class_idx in range(n_classes):
        y_true_binary = y_true_onehot[:, class_idx]
        y_score_binary = y_score[:, class_idx]
        
        if torch.sum(y_true_binary == 1) == 0 or torch.sum(y_true_binary == 0) == 0:
            auc = torch.tensor(0.5, device=device)
        else:
            auc = binary_roc_auc(y_true_binary, y_score_binary)
        auc_scores.append(auc)
        valid_classes.append(class_idx)
    
    if len(auc_scores) == 0:
        return torch.tensor(0.5, device=device)  
    
    auc_scores = torch.stack(auc_scores)
    
    if average == 'macro':
        return torch.mean(auc_scores)
    elif average == 'micro':
        y_true_flat = y_true_onehot.flatten()
        y_score_flat = y_score.flatten()
        return binary_roc_auc(y_true_flat, y_score_flat)
    elif average == 'weighted':
        weights = torch.tensor([torch.sum(y_true_onehot[:, i]) 
                               for i in valid_classes], device=device)
        weights = weights / torch.sum(weights)
        return torch.sum(auc_scores * weights)
    else:
        return auc_scores

def binary_roc_auc(y_true, y_score):
    device = y_true.device
    
    sorted_indices = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[sorted_indices]
    
    n_pos = torch.sum(y_true == 1).float()
    n_neg = torch.sum(y_true == 0).float()
    
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.5, device=device)
    
    tp = torch.cumsum(y_true_sorted, dim=0).float()
    fp = torch.cumsum(1 - y_true_sorted, dim=0).float()
    
    tpr = tp / n_pos
    fpr = fp / n_neg
    
    tpr = torch.cat([torch.tensor([0.0], device=device), tpr])
    fpr = torch.cat([torch.tensor([0.0], device=device), fpr])
    
    auc = torch.trapz(tpr, fpr)
    
    return auc

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     maxk = min(max(topk), output.size()[1])
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.reshape(1, -1).expand_as(pred))
#     return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

class ParticipantVisibleError(Exception):
    pass

def accuracy(
    y_scores: torch.Tensor,
    y_true: torch.Tensor,
    class_weights: Optional[List[float]] = None,
    if_final: bool = False,
    # threshold: float = 0.5,
) -> float:
    """Compute weighted AUC for multilabel classification.

    Parameters:
    -----------
    y_scores : np.ndarray of shape (n_samples, n_classes)
        Target scores (probability estimates or decision values)
    y_true : np.ndarray of shape (n_samples, n_classes)
        True binary labels (0 or 1) for each class
    class_weights : array-like of shape (n_classes,), optional
        Weights for each class. If None, uniform weights are used.
        Weights will be normalized to sum to 1.

    Returns:
    --------
    weighted_auc : float
        The weighted average AUC

    Raises:
    -------
    ValueError
        If any class does not have both positive and negative samples
    """
    n_classes = y_true.shape[1]

    # for i in range(len(y_scores)):
    #     if torch.any(y_scores[i, :-1] > threshold): 
    #         y_scores[i, -1] = 1  
        
    try:
        y_scores = torch.nan_to_num(y_scores,nan=0.5)
        individual_aucs = roc_auc_score(y_true, y_scores, average=None)
        individual_aucs = torch.nan_to_num(individual_aucs,nan=0)
        # individual_accuracies = (y_true == (y_scores>threshold)).astype(int)
        # individual_aucs = np.mean(individual_accuracies, axis=0)
    except ValueError:
        raise ParticipantVisibleError(
            'AUC could not be calculated from given predictions.'
        ) from None

    # Handle weights
    if class_weights is None:  # Uniform weights
        weights_array = torch.ones(n_classes,device=y_true.device)
        if if_final:
            weights_array[-1] = 13
    else:
        weights_array = class_weights

    # Check weight dimensions
    if len(weights_array) != n_classes:
        raise ValueError(
            f'Number of weights ({len(weights_array)}) must match '
            f'number of classes ({n_classes})'
        )

    # Check for non-negative weights
    if torch.any(weights_array < 0):
        raise ValueError('All class weights must be non-negative')

    # Check that at least one weight is positive
    if torch.sum(weights_array) == 0:
        raise ValueError('At least one class weight must be positive')

    # Normalize weights to sum to 1
    weights_array = weights_array / torch.sum(weights_array)
    # Compute weighted average
    return torch.sum(individual_aucs * weights_array), individual_aucs
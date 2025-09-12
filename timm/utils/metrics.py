""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import tensor
from typing import List, Optional

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
    y_scores: np.ndarray,
    y_true: np.ndarray,
    class_weights: Optional[List[float]] = None,
    threshold: float = 0.5,
) -> float:
    """Compute weighted AUC for multilabel classification.

    Parameters:
    -----------
    y_true : np.ndarray of shape (n_samples, n_classes)
        True binary labels (0 or 1) for each class
    y_scores : np.ndarray of shape (n_samples, n_classes)
        Target scores (probability estimates or decision values)
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
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_true.shape[1]

    for i in range(len(y_scores)):
        if np.any(y_scores[i, :-1] > threshold):  # 如果前13个标签中有一个大于threshold
            y_scores[i, -1] = 1  # 设置第14个标签为1
        
    # Get AUC for each class
    try:
        y_scores = np.nan_to_num(y_scores,nan=0)
        individual_aucs = roc_auc_score(y_true, y_scores, average=None)
        individual_aucs = np.nan_to_num(individual_aucs,nan=0)
        # individual_accuracies = (y_true == (y_scores>threshold)).astype(int)
        # individual_aucs = np.mean(individual_accuracies, axis=0)
    except ValueError:
        raise ParticipantVisibleError(
            'AUC could not be calculated from given predictions.'
        ) from None

    # Handle weights
    if class_weights is None:  # Uniform weights
        weights_array = np.ones(n_classes)
        weights_array[-1] = 13
    else:
        weights_array = np.asarray(class_weights)

    # Check weight dimensions
    if len(weights_array) != n_classes:
        raise ValueError(
            f'Number of weights ({len(weights_array)}) must match '
            f'number of classes ({n_classes})'
        )

    # Check for non-negative weights
    if np.any(weights_array < 0):
        raise ValueError('All class weights must be non-negative')

    # Check that at least one weight is positive
    if np.sum(weights_array) == 0:
        raise ValueError('At least one class weight must be positive')

    # Normalize weights to sum to 1
    weights_array = weights_array / np.sum(weights_array)

    # Compute weighted average
    return tensor(np.sum(individual_aucs * weights_array)), tensor(individual_aucs)
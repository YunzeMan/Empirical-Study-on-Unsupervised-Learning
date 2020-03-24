from time import time
import numpy as np


import torch
import torchvision
from torchvision import datasets, transforms

# some utils


def assign_labels_to_centroids(pred_labels, gt_labels):
    n_digits = len(np.unique(gt_labels))
    assigned_labels = []
    final_pred = np.zeros_like(gt_labels)
    for i in range(n_digits):
        mask = (pred_labels == i)
        most_freq_label = np.argmax(np.bincount(gt_labels[mask]))
        assigned_labels.append(most_freq_label)
        final_pred[mask] = most_freq_label

    return assigned_labels, final_pred


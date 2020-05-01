from time import time
import numpy as np
from scipy.optimize import linear_sum_assignment

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

def assign_labels_to_centroids_bipartite(pred_labels, gt_labels):
    n_digits = len(np.unique(gt_labels))
    assigned_labels = []
    final_pred = np.zeros_like(gt_labels)

    cost_matrix = np.zeros((n_digits, n_digits))

    for i in range(n_digits):
        mask = (pred_labels == i)
        cost_matrix[i, :] = np.bincount(gt_labels[mask], minlength=np.max(gt_labels)+1) # take minus here, low cost for better match
        cost_matrix[i, :] = -cost_matrix[i, :] / np.sum(cost_matrix[i, :])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_labels = col_ind

    for i in range(n_digits):
        mask = (pred_labels == i)
        final_pred[mask] = assigned_labels[i]

    return assigned_labels, final_pred


from time import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import torch
import torchvision
from torchvision import datasets, transforms

# some utils

def get_all_data_mnist(dataset):
    n_samples = len(dataset)
    img = torch.zeros(n_samples, 28, 28)
    label = torch.zeros(n_samples)

    for i, (data, target) in enumerate(dataset):
        img[i] = data
        label[i] = target

    return img.numpy(), label.numpy()

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



mnist_trainset = datasets.MNIST(root='../dataset', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='../dataset', train=False, download=True, transform=transforms.ToTensor())

train_data, train_label = get_all_data_mnist(mnist_trainset) # train_data is 60000 x 28 x 28

n_samples = train_data.shape[0]
n_digits = len(np.unique(train_label))

print("n_digits: %d" % n_digits)
print("n_samples: %d" % n_samples)

train_data = train_data.reshape(-1, 784) # flatten, 60000 x 784
train_label = train_label.astype(int)

test_data, test_label = get_all_data_mnist(mnist_testset)
test_data = test_data.reshape(-1, 784)
test_label = test_label.astype(int)


print("Raw k-means with k-means++ init...")
t0 = time()
km = KMeans(init='k-means++', n_clusters=n_digits, n_init=1).fit(train_data)
print("done in %0.3fs" % (time() - t0))
_, train_final_pred = assign_labels_to_centroids(km.labels_, train_label)
train_acc = np.sum(train_final_pred == train_label) / n_samples
print("train acc: %.3f" % train_acc)

_, test_final_pred = assign_labels_to_centroids(km.predict(test_data), test_label)
test_acc = np.sum(test_final_pred == test_label) / len(test_label)
print("test acc: %.3f" % test_acc)



print("k-means with pca...")
t0 = time()
pca = PCA(n_components=20).fit(train_data)
train_data_pca = pca.fit_transform(train_data) # 60000 x 50
km = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(train_data_pca)
print("done in %0.3fs" % (time() - t0))
_, train_final_pred = assign_labels_to_centroids(km.labels_, train_label)
train_acc = np.sum(train_final_pred == train_label) / n_samples
print("train acc: %.3f" % train_acc)

_, test_final_pred = assign_labels_to_centroids(km.predict(pca.transform(test_data)), test_label)
test_acc = np.sum(test_final_pred == test_label) / len(test_label)
print("test acc: %.3f" % test_acc)

# print(len(mnist_trainset))
# print(np.asarray(mnist_trainset[0]))

# digits, labels = load_digits(return_X_y=True)

# n_samples, n_features = digits.shape
# n_digits = len(np.unique(labels))

# print("n_digits: %d" % n_digits)
# print("n_features: %d" % n_features)
# print("n_samples: %d" % n_samples)

# print "n_digits: %d" % n_digits
# print "n_features: %d" % n_features
# print "n_samples: %d" % n_samples
# print

# print "Raw k-means with k-means++ init..."
# t0 = time()
# km = KMeans(init='k-means++', k=n_digits, n_init=10).fit(data)
# print "done in %0.3fs" % (time() - t0)
# print "inertia: %f" % km.inertia_
# print

# print "Raw k-means with random centroid init..."
# t0 = time()
# km = KMeans(init='random', k=n_digits, n_init=10).fit(data)
# print "done in %0.3fs" % (time() - t0)
# print "inertia: %f" % km.inertia_
# print

# print "Raw k-means with PCA-based centroid init..."
# # in this case the seeding of the centers is deterministic, hence we run the
# # kmeans algorithm only once with n_init=1
# t0 = time()
# pca = PCA(n_components=n_digits).fit(data)
# km = KMeans(init=pca.components_.T, k=n_digits, n_init=1).fit(data)
# print "done in %0.3fs" % (time() - t0)
# print "inertia: %f" % km.inertia_
# print
from time import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import torch
import torchvision
from torchvision import datasets, transforms
from lib.dataset_shapenet import ShapenetDataset

def get_all_data_mnist(dataset):
    n_samples = len(dataset)
    img = torch.zeros(n_samples, 28, 28)
    label = torch.zeros(n_samples)

    for i, (data, target) in enumerate(dataset):
        img[i] = data
        label[i] = target
    return img.numpy(), label.numpy()

def prepare_mnist_dataset():
    print("************************ Loading mnist ************************")
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

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
    print("************************ Finished mnist ************************")
    return train_data, train_label, test_data, test_label



def get_all_data_shapenet(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1200,  shuffle=True, num_workers=8)
    imgs = []
    targets = []

    for i, (data, target) in enumerate(loader):
        imgs.append(data.numpy().squeeze())
        targets.append(target.numpy().squeeze())
    imgs = np.concatenate(imgs)
    targets = np.concatenate(targets)
    return imgs.reshape(imgs.shape[0], -1), targets.astype(int)

def prepare_shapenet_dataset(shapenet_root):
    transforms_shapenet = transforms.Compose([transforms.Grayscale(), transforms.Resize(128), transforms.ToTensor()])
    train_dataset = ShapenetDataset("train", shapenet_root, transforms=transforms_shapenet)
    test_dataset = ShapenetDataset("test", shapenet_root, transforms=transforms_shapenet)

    train_imgs, train_labels = get_all_data_shapenet(train_dataset)
    test_imgs, test_labels = get_all_data_shapenet(test_dataset)
    return train_imgs, train_labels, test_imgs, test_labels


def assign_majority(pred, label):
    unique_pred = np.unique(pred)
    final_pred = np.zeros(pred.shape)
    label_dict = {}

    for i in unique_pred:
        pred_cases = np.where(pred == i)
        case_labels = label[pred_cases[0]]
        counts, freq = np.unique(case_labels, return_counts=True)
        assigned_label = counts[np.argmax(freq)]
        label_dict[i] = assigned_label
    for i in unique_pred:
        final_pred[pred == i] = label_dict[i]
    return final_pred, label_dict


def pred_majority(pred, label_dict):
    unique_pred = np.unique(pred)
    final_pred = np.zeros(pred.shape)
    for i in unique_pred:
        final_pred[pred == i] = label_dict[i]
        
    return final_pred

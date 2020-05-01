import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import numpy as np
import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import mixture

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from network import Net_MNIST, Net_ShapeNet

from libs.kmeans_lib import *
from libs.utils import *
from libs.dataset_shapenet import ShapenetDataset



def visualize(mnist, targets, ret):
    target_ids = range(len(set(targets)))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']
    
    plt.figure(figsize=(12, 10))
    
    ax = plt.subplot(aspect='equal')
    for label in set(targets):
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label], label=label)
    
    for i in range(0, len(targets), 250):
        img = (mnist[i][0] * 0.3081 + 0.1307).numpy()[0]
        img = OffsetImage(img, cmap=plt.cm.gray_r, zoom=0.5) 
        ax.add_artist(AnnotationBbox(img, ret[i]))
    
    plt.legend()
    plt.savefig('fig.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo example: MNIST')
    parser.add_argument('--queue', '-q', type=int, default=128)
    parser.add_argument('--beta', '-b', type=float, default=0.999)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--dataset', '-d', type=str, default='ShapeNet')
    parser.add_argument('--dataset_dir', '-dir', type=str, default='/home/yunze/dataset')

    args = parser.parse_args()

    if args.dataset == 'MNIST':
        model_path = 'result/mnist/model_%d_%.4f_epoch%d.pth' %(args.queue, args.beta, args.epoch)
    else:
        model_path = 'result/chair_cls1/model_%d_%.4f_epoch%d.pth' %(args.queue, args.beta, args.epoch)
        dataset_dir = os.path.join(args.dataset_dir, 'chair_cls1')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    if args.dataset == 'MNIST':
        train_dataset = datasets.MNIST('./', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./', train=False, download=True, transform=transform)
    else:
        train_dataset = ShapenetDataset(mode='train', dataset_root=dataset_dir, transform=transform)
        test_dataset = ShapenetDataset(mode='test', dataset_root=dataset_dir, transform=transform)

    if args.dataset == 'MNIST':
        Net = Net_MNIST
    else:
        Net = Net_ShapeNet

    model = Net(args.queue)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    train_data = []
    train_label = []
    for m in tqdm.tqdm(train_dataset):
        target = m[1]
        train_label.append(target)
        x = m[0]
        x = x.view(1, *x.shape)
        feat = model(x)
        train_data.append(feat.data.numpy()[0])

    test_data = []
    test_label = []
    for m in tqdm.tqdm(test_dataset):
        target = m[1]
        test_label.append(target)
        x = m[0]
        x = x.view(1, *x.shape)
        feat = model(x)
        test_data.append(feat.data.numpy()[0])

    train_data = np.array(train_data)
    train_label = np.array(train_label)

    test_data = np.array(test_data)
    test_label = np.array(test_label)

    print(train_data.shape, train_label.shape)
    # ret = TSNE(n_components=2, random_state=0).fit_transform(test_data)
   
    # visualize(test_dataset, test_label, ret)
    train_data_pca = train_data
    test_data_pca = test_data

    cov_type = 'full'
    n_comp = 130
    print('The length of queue is %d, the momentum beta is %.4f, the number of epoch is %d' %(args.queue, args.beta, args.epoch), file=open("output.txt", "a"))
    print("Train:", train_data_pca.shape, "Test:", test_data_pca.shape, "Diag_type:", cov_type, "Num_comp", n_comp)
    gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type=cov_type)
    gmm.fit(train_data_pca)

    train_pred = gmm.predict(train_data_pca)
    train_final_pred, label_dict = assign_majority(train_pred, train_label)
    print("Train Acc:", np.mean(train_final_pred == train_label), file=open("output.txt", "a"))

    test_pred = gmm.predict(test_data_pca)
    test_final_pred = pred_majority(test_pred, label_dict)
    print("Test Acc:", np.mean(test_final_pred == test_label), file=open("output.txt", "a"))
    print("", file=open("output.txt", "a"))
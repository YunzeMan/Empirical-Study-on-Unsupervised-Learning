import argparse
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
from network import Net

from libs.kmeans_lib import *
from libs.utils import *


def show(mnist, targets, ret):
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
    test_method = 2
    parser = argparse.ArgumentParser(description='MoCo example: MNIST')
    parser.add_argument('--queue', '-q', type=int, default=128)
    parser.add_argument('--beta', '-b', type=float, default=0.999)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    args = parser.parse_args()

    model_path = 'result/model_%d_%.4f_epoch%d.pth' %(args.queue, args.beta, args.epoch)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = datasets.MNIST('./', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./', train=False, download=True, transform=transform)
    
    model = Net(args.queue)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    train_data = []
    train_label = []
    for m in tqdm.tqdm(mnist_train):
        target = m[1]
        train_label.append(target)
        x = m[0]
        x = x.view(1, *x.shape)
        feat = model(x)
        train_data.append(feat.data.numpy()[0])

    test_data = []
    test_label = []
    for m in tqdm.tqdm(mnist_test):
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
   
    # show(mnist_test, test_label, ret)
    if test_method == 1:
        # pca = PCA(0.5, whiten=True)
        # pca = pca.fit(train_data)
        # train_data_pca = pca.transform(train_data)
        # test_data_pca = pca.transform(test_data)

        train_data_pca = train_data
        test_data_pca = test_data

        cov_type = 'full'
        n_comp = 80
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
    else:
        print("Raw k-means with k-means++ init...")
        t0 = time()
        km = KMeans(init='k-means++', n_clusters=10, n_init=10).fit(train_data)
        print("done in %0.3fs" % (time() - t0))

        assigned_labels, train_final_pred = assign_majority(km.labels_, train_label)
        train_acc = np.mean(train_final_pred == train_label)
        print("train acc: %.3f" % train_acc)

        predicted_label = km.predict(test_data)
        test_final_pred = np.zeros_like(predicted_label)
        for i in range(10):
            test_final_pred[predicted_label == i] = assigned_labels[i]

        # _, test_final_pred = assign_labels_to_centroids(km.predict(test_data), test_label)
        test_acc = np.mean(test_final_pred == test_label)
        print("test acc: %.3f" % test_acc)

        print("k-means with pca...")
        t0 = time()
        pca = PCA(n_components=20).fit(train_data)
        train_data_pca = pca.fit_transform(train_data) # 60000 x 50
        km = KMeans(init='k-means++', n_clusters=10, n_init=10).fit(train_data_pca)
        print("done in %0.3fs" % (time() - t0))
        assigned_labels, train_final_pred = assign_labels_to_centroids_bipartite(km.labels_, train_label)
        train_acc = np.mean(train_final_pred == train_label)
        print("train acc: %.3f" % train_acc)

        predicted_label = km.predict(pca.transform(test_data))
        test_final_pred = np.zeros_like(predicted_label)
        for i in range(10):
            test_final_pred[predicted_label == i] = assigned_labels[i]

        test_acc = np.mean(test_final_pred == test_label)
        print("test acc: %.3f" % test_acc)


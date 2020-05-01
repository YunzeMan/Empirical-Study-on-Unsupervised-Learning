import os
import sys
sys.path.append(os.getcwd())

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from lib.kmeans_lib import *
from lib.utils import *


if __name__ == "__main__":
    dataset = 'mnist'
    dataset = 'shapenet'

    if dataset == 'mnist':
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

    if dataset == 'shapenet':
        train_data, train_label, test_data, test_label = prepare_shapenet_dataset("./chair_cls1")

        n_digits=10
        n_samples = 60000


    # print("Raw k-means with k-means++ init...")
    # t0 = time()
    # km = KMeans(init='k-means++', n_clusters=n_digits, n_init=1).fit(train_data)
    # print("done in %0.3fs" % (time() - t0))
    # assigned_labels, train_final_pred = assign_labels_to_centroids_bipartite(km.labels_, train_label)
    # print(train_label)
    # print(assigned_labels)
    # print(train_final_pred)
    # train_acc = np.sum(train_final_pred == train_label) / n_samples
    # print("train acc: %.3f" % train_acc)
    # assigned_labels, train_final_pred = assign_majority(km.labels_, train_label)
    # print(train_final_pred)
    # print(assigned_labels)
    # train_acc = np.sum(assigned_labels == train_label) / n_samples
    # print("train acc: %.3f" % train_acc)

    # predicted_label = km.predict(test_data)
    # test_final_pred = np.zeros_like(predicted_label)
    # for i in range(10):
    #     test_final_pred[predicted_label == i] = assigned_labels[i]

    # # _, test_final_pred = assign_labels_to_centroids(km.predict(test_data), test_label)
    # test_acc = np.sum(test_final_pred == test_label) / len(test_label)
    # print("test acc: %.3f" % test_acc)

    print("k-means with pca...")
    t0 = time()
    pca = PCA(n_components=50).fit(train_data)
    train_data_pca = pca.fit_transform(train_data) # 60000 x 50
    print("done pca...")
    km = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit(train_data_pca)
    print("done in %0.3fs" % (time() - t0))
    assigned_labels, train_final_pred = assign_labels_to_centroids_bipartite(km.labels_, train_label)
    train_acc = np.sum(train_final_pred == train_label) / n_samples
    print("train acc: %.3f" % train_acc)

    predicted_label = km.predict(pca.transform(test_data))
    test_final_pred = np.zeros_like(predicted_label)
    for i in range(10):
        test_final_pred[predicted_label == i] = assigned_labels[i]

    test_acc = np.sum(test_final_pred == test_label) / len(test_label)
    print("test acc: %.3f" % test_acc)
import os
import sys
sys.path.append(os.getcwd())

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import mixture
import matplotlib.pyplot as plt

from lib.utils import *

if __name__ == "__main__":
    
    # train_data, train_label, test_data, test_label = prepare_mnist_dataset()
    train_data, train_label, test_data, test_label = prepare_shapenet_dataset("/hdd/zen/data/Reallite/Rendering/chair_cls1")
    
    pca = PCA(0.7, whiten=True)
    pca = pca.fit(train_data)
    train_data_pca = pca.transform(train_data)
    test_data_pca = pca.transform(test_data)

    cov_type = 'full'
    n_comp = 80
    print("Train:", train_data_pca.shape, "Test:", test_data_pca.shape, "Diag_type:", cov_type, "Num_comp", n_comp)
    gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type=cov_type)
    gmm.fit(train_data_pca)

    train_pred = gmm.predict(train_data_pca)
    train_final_pred, label_dict = assign_majority(train_pred, train_label)
    print("Train Acc:", np.mean(train_final_pred == train_label))

    test_pred = gmm.predict(test_data_pca)
    test_final_pred = pred_majority(test_pred, label_dict)
    print("Test Acc:", np.mean(test_final_pred == test_label))
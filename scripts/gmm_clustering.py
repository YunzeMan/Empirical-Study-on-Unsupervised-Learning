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
    
    train_data, train_label, test_data, test_label =  prepare_mnist_dataset()
    pca = PCA(0.99, whiten=True)
    data = pca.fit_transform(train_data)
    print(data.shape)
    n_components = np.arange(50, 210, 10)
    # models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0)
    #         for n in n_components]
    # aics = []
    # for model in models:
    #     print("Fitting for model:", model)
    #     aic = model.fit(data).aic(data) 
    # plt.plot(n_components, aics)
    print("Fitting GMM")
    print(train_data.shape)
    g = mixture.GaussianMixture(n_components=10)
    g.fit(train_data) 

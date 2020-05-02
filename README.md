# Unsupervised-classification-of-datasets
10-701 Course Project 

[Midway report overleaf link](https://www.overleaf.com/4123148767nnmknvwnmtry)

# Overview
In this project, we studies the problem of unsupervised visual learning. We investigate both classical and state of the art unsupervsied learning methods and compare their performance. To analyze these methods, we choose the task of image classifcation. We run our experiments on the popular MNIST dataset, and also extend our experiments to a self-generated 3D shape dataset. 

# Folder Structure
    .
    ├── Capsules                   # Folder contains experiments about Stacked Capsule Autoencoders
    ├── MoCo                       # Folder contains experiments about MoCo (Momentum Contrast)
    ├── Autoencoder                # Folder contains experiments about Autoencoder    
    ├── InfoGAN                    # Folder contains experiments about InfoGAN
    ├── scripts                    # Scripts to run different baseline methods such as GMM & K-Means
    └── README.md
    

# Environment
To install the repective environment, refer to [Capsules](Capsules/README.md) and [MoCo](MoCo/README.md) for installation details. 

# Dataset
Our experiments are done on MNIST & self-rendered shapenet dataset. The shapeNet dataset can be downloaded [here](https://drive.google.com/file/d/1msGsrX48YB92bm2f1YgxESuddvGu2Ywt/view?usp=sharing).


# Run baseline methods (GMM, K-means)

To run baseline methods, use the following commands:

```bash 
python scripts/gmm_clustering.py --dataset mnist
```

To run shapeNet tests, remember to change the the dataset directory to your extracted dataset directory. 

```bash 
python scripts/gmm_clustering.py --dataset shapenet
```

```bash 
python scripts/kmeans_clustering.py --dataset mnist
```

# Run MoCo, Capsule, VAE experiments
Please refer to [Capsules](Capsules/README.md),  [MoCo](MoCo/README.md) and [Autoencoder](Autoencoder/README.md) for training and testing details.

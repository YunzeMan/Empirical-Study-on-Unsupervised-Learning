import os
import sys
sys.path.append(os.getcwd())

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from sklearn.cluster import KMeans
from sklearn import mixture
from lib.kmeans_lib import *
from lib.utils import *

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)


class autoencoder(nn.Module):
    def __init__(self, emb_dim):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, emb_dim))
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        feature = self.encoder(x)
        x = self.decoder(feature)
        return x, feature

emb_dim = 12

model = autoencoder(emb_dim).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

# mode = 'train'
mode = 'test'

# backend = 'kmeans'
backend = 'gmm'

if mode == 'train':
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            output, _ = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './sim_autoencoder.pth')

elif mode == 'test':
    ckpt = torch.load('./sim_autoencoder.pth')
    model.load_state_dict(ckpt)
    train_features = np.zeros((len(mnist_trainset), emb_dim))
    test_features = np.zeros((len(mnist_testset), emb_dim))
    train_labels = np.zeros(len(mnist_trainset))
    test_labels = np.zeros(len(mnist_testset))

    cur_pos = 0

    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        _, feature = model(img)
        
        train_features[cur_pos:cur_pos+len(img), :] = feature.detach().cpu().numpy()
        train_labels[cur_pos:cur_pos+len(img)] = label.numpy().astype(np.int32)
        cur_pos += len(img)

    cur_pos = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        _, feature = model(img)
        
        test_features[cur_pos:cur_pos+len(img), :] = feature.detach().cpu().numpy()
        test_labels[cur_pos:cur_pos+len(img)] = label.numpy().astype(np.int32)
        cur_pos += len(img)

    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    print("done extracting features ... ")

    if backend == 'kmeans':
        km = KMeans(init='k-means++', n_clusters=10, n_init=10).fit(train_features)
        assigned_labels, train_final_pred = assign_labels_to_centroids_bipartite(km.labels_, train_labels)
        train_acc = np.sum(train_final_pred == train_labels) / len(train_labels)
        print("train acc: %.3f" % train_acc)

        predicted_label = km.predict(test_features)
        test_final_pred = np.zeros_like(predicted_label)
        for i in range(10):
            test_final_pred[predicted_label == i] = assigned_labels[i]

        test_acc = np.sum(test_final_pred == test_labels) / len(test_labels)
        print("test acc: %.3f" % test_acc)

    elif backend == 'gmm':
        cov_type = 'full'
        n_comp = 40
        print("Train:", train_features.shape, "Test:", test_features.shape, "Diag_type:", cov_type, "Num_comp", n_comp)
        gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type=cov_type)
        gmm.fit(train_features)

        train_pred = gmm.predict(train_features)
        train_final_pred, label_dict = assign_majority(train_pred, train_labels)
        print("Train Acc:", np.mean(train_final_pred == train_labels))

        test_pred = gmm.predict(test_features)
        test_final_pred = pred_majority(test_pred, label_dict)
        print("Test Acc:", np.mean(test_final_pred == test_labels))

    

    




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
from lib.dataset_shapenet import ShapenetDataset
from lib.models import SegNet

# if not os.path.exists('./mlp_img'):
#     os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

img_size = 64
feature_size = 200
num_epochs = 10
batch_size = 32
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

transforms_shapenet = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor()])
mnist_trainset = ShapenetDataset("train", "./chair_cls1", transforms=transforms_shapenet)
mnist_testset = ShapenetDataset("test", "./chair_cls1", transforms=transforms_shapenet)

train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)


class autoencoder(nn.Module):
    def __init__(self, emb_dim):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, emb_dim, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        feature = self.encoder(x)
        x = self.decoder(feature)
        return x, feature.view(feature.size(0), -1)


emb_dim = 8

model = autoencoder(emb_dim).cuda()
# model = SegNet(n_classes=1, in_channels=1).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

mode = 'train'
# mode = 'test'

if mode == 'train':

    for epoch in range(num_epochs):
        i = 1
        for data in train_loader:
            img, _ = data
            # img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            output, feat = model(img)
            # print(feat.shape)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('finish step {}/{}, loss = {}'.format(i, len(train_loader), loss.data))
            i += 1
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data))
        # if epoch % 10 == 0:
        #     pic = to_img(output.cpu().data)
        #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')

elif mode == 'test':
    ckpt = torch.load('./conv_autoencoder.pth')
    model.load_state_dict(ckpt)
    train_features = np.zeros((len(mnist_trainset), feature_size))
    test_features = np.zeros((len(mnist_testset), feature_size))
    train_labels = np.zeros(len(mnist_trainset))
    test_labels = np.zeros(len(mnist_testset))

    cur_pos = 0
    i = 1
    for data in train_loader:
        img, label = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output, feature = model(img)

        print(criterion(output, img))
        
        train_features[cur_pos:cur_pos+len(img), :] = feature.detach().cpu().numpy()
        train_labels[cur_pos:cur_pos+len(img)] = label.numpy().astype(np.int32)
        cur_pos += len(img)

        print('finish trainset step {}/{}'.format(i, len(train_loader)))
        i += 1

    torch.save([train_features, train_labels], 'conv_ae_train_feat.pt')

    cur_pos = 0
    i = 1
    for data in test_loader:
        img, label = data
        img = Variable(img).cuda()
        # ===================forward=====================
        _, feature = model(img)
        
        test_features[cur_pos:cur_pos+len(img), :] = feature.detach().cpu().numpy()
        test_labels[cur_pos:cur_pos+len(img)] = label.numpy().astype(np.int32)
        cur_pos += len(img)
        print('finish testset step {}/{}'.format(i, len(test_loader)))
        i += 1

    torch.save([test_features, test_labels], 'conv_ae_test_feat.pt')

    train_features, train_labels = torch.load('conv_ae_train_feat.pt')
    test_features, test_labels = torch.load('conv_ae_test_feat.pt')

    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    print('doing pca...')

    pca = PCA(50, whiten=True)
    pca = pca.fit(train_features)
    train_features= pca.transform(train_features)
    test_features = pca.transform(test_features)

    # gmm

    cov_type = 'full'
    n_comp = 130
    print("Train:", train_features.shape, "Test:", test_features.shape, "Diag_type:", cov_type, "Num_comp", n_comp)
    gmm = mixture.GaussianMixture(n_components=n_comp, covariance_type=cov_type)
    gmm.fit(train_features)

    train_pred = gmm.predict(train_features)
    train_final_pred, label_dict = assign_majority(train_pred, train_labels)
    print("Train Acc:", np.mean(train_final_pred == train_labels))

    test_pred = gmm.predict(test_features)
    test_final_pred = pred_majority(test_pred, label_dict)
    print("Test Acc:", np.mean(test_final_pred == test_labels))
    

    




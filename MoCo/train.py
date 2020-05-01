import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import copy
from libs.dataset_shapenet import ShapenetDataset

from network import Net_MNIST, Net_ShapeNet

class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2

def momentum_update(model_q, model_k, beta = 0.999):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
    model_k.load_state_dict(param_k)

def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=4096):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def initialize_queue(model_k, device, train_loader, queue_length=128):
    queue = torch.zeros((0, queue_length), dtype=torch.float) 
    queue = queue.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.to(device)
        k = model_k(x_k)
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K = 10)
        break
    return queue

def train(model_q, model_k, device, train_loader, queue, optimizer, epoch, temp=0.07, beta=0.999):
    model_q.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        x_q = data[0]
        x_k = data[1]

        x_q, x_k = x_q.to(device), x_k.to(device)
        q = model_q(x_q)
        k = model_k(x_k)
        k = k.detach()

        N = data[0].shape[0]
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N,1,-1), k.view(N,-1,1))
        l_neg = torch.mm(q.view(N,-1), queue.T.view(-1,K))

        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long)
        labels = labels.to(device)

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits/temp, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        momentum_update(model_q, model_k, beta=beta)

        queue = queue_data(queue, k)
        queue = dequeue_data(queue)

    total_loss /= len(train_loader.dataset)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo example: MNIST')
    parser.add_argument('--batchsize', '-bs', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--dataset', '-d', type=str, default='MNIST',
                        help='MNIST or ShapeNet')
    parser.add_argument('--dataset_dir', '-dir', type=str, default='/home/yunze/dataset',
                        help='The dir of dataset')
    parser.add_argument('--epochs', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--queue', '-q', type=int, default=128,
                        help='Queue length')
    parser.add_argument('--beta', '-b', type=float, default=0.999,
                        help='Beta')

    args = parser.parse_args()

    beta = args.beta
    queue_length = args.queue

    dataset = args.dataset
    batchsize = args.batchsize
    epochs = args.epochs

    if dataset == 'MNIST':
        dataset_dir = os.path.join(args.dataset_dir, 'mnist')
        out_dir = os.path.join(args.out, 'mnist')
    else:
        dataset_dir = os.path.join(args.dataset_dir, 'chair_cls1')
        out_dir = os.path.join(args.out, 'chair_cls1')

    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    if dataset == 'MNIST':
        transform = DuplicatedCompose([
            transforms.RandomRotation(20, fill=(0,)),
            transforms.RandomResizedCrop(28, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transform = DuplicatedCompose([
            transforms.RandomRotation(20, fill=(0,)),
            transforms.RandomResizedCrop(112, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    
    if dataset == 'MNIST':
        train_dataset = datasets.MNIST('./', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./', train=False, download=True, transform=transform)
    else:
        train_dataset = ShapenetDataset(mode='train', dataset_root=dataset_dir, transform=transform)
        test_dataset = ShapenetDataset(mode='test', dataset_root=dataset_dir, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    
    if dataset == 'MNIST':
        Net = Net_MNIST
    else:
        Net = Net_ShapeNet
    
    model_q = Net(queue_length).to(device)
    model_k = copy.deepcopy(model_q)
    optimizer = optim.SGD(model_q.parameters(), lr=0.01, weight_decay=0.0001)
    
    
    queue = initialize_queue(model_k, device, train_loader, queue_length=queue_length)
   
    for epoch in range(1, epochs + 1):
        train(model_q, model_k, device, train_loader, queue, optimizer, epoch, beta=beta)
        if epoch % 50 == 0:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model_q.state_dict(), os.path.join(out_dir, 'model_%d_%.4f_epoch%d.pth' %(queue_length, beta, epoch)))

    

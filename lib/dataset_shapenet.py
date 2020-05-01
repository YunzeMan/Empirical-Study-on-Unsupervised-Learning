import os
import sys
import pdb
import glob
sys.path.append(os.getcwd())

from PIL import Image
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import time
import random
# import numpy.ma as ma


input_size = 128


class ShapenetDataset(data.Dataset):
    def __init__(self, mode, dataset_root, rot_repr = "quat",  transforms=None, shuffle = True, split = 6/7):
        print("********************** Loading Shapenet Dataset **********************")
        np.random.seed(1) # train test split need to stablize
        self.transforms = transforms
        self.root = dataset_root
        self.parse_dataset()
        if shuffle:
            np.random.shuffle(self.data_list)
        
        if mode == 'train':
            split_num = np.floor(split * len(self.data_list)).astype(int)
            self.data_list = self.data_list[:split_num]
        elif mode == 'test':
            split_num = np.floor((1- split) * len(self.data_list)).astype(int)
            self.data_list = self.data_list[:split_num]

        
        self.length = len(self.data_list)
        
        print("Dataset Mode: ", mode)
        print("Dataset Length: ", len(self.data_list))
        print("********************** Finished Shapenet Dataset **********************")

    def parse_dataset(self, num_classes = 10):
        # classes = sorted(os.listdir(os.path.join(self.root, "data")))
        # classes = np.random.choice(classes, num_classes, replace = False)
        classes = ['38436cce91dfe9a340b2974a4bd47901', 'ccfc857f35c138ede785b88cc9024b2a',
                    '3fdc09d3065fa3c524e7e8a625efb2a7', '795f38ce5d8519938077cafed2bb8242',
                    '30b0196b3b5431da2f95e2a1e9997b85', 'b405fa036403fbdb307776da88d1350f',
                    '3af3096611c8eb363d658c402d71b967', 'ce8e6c13899376e2f3c9c1464e55d580',
                    '4527cc19d2f09853a718067b9ac932e1', '124ef426dfa0aa38ff6069724068a578']
        # print(classes)
        self.classes = classes
        self.class_2_label = { classes[i]:i for i in range(len(classes))}
        
        all_images = []
        for class_name in classes:
            all_image = sorted(glob.glob(os.path.join(self.root, "data", class_name, "*-color.png")))
            all_image = [os.path.join(self.root, "data", class_name, i.split("/")[-1].split("-")[0])  for i in all_image]
            all_images = all_images + all_image

        self.data_list = all_images
        
        
        
    def __getitem__(self, index):

        try:
            img = Image.open('{}-color.png'.format( self.data_list[index]))
            meta = np.load('{}-meta.npz'.format(self.data_list[index]))
        except Exception as e:
            print(e)
            print(self.data_list[index])
#         img = np.array(img)
        model_id = str(meta['model_id'])
        label = self.class_2_label[model_id]
        
        if self.transforms:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.length
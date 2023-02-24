# DATALOADER
import os
import pandas as pd
from torchvision.io import read_image
from pathlib import Path
import torch
import os.path
import cv2
import scipy.ndimage
import random
import numpy as np
from skimage import filters
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from meshobj import MeshObj

max_depth_range = 2.1
mesh_data_root = 'demodata'
binvoxPathPrefix = 'demodata'
extra_mesh_data_root = None


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, n_planes_for_train):
        # making dataframe 
        self.df2 = pd.read_csv(csv_file) 
        headerList = ['image', 'visible', 'depth', 'gt_oplane']
        self.df2.to_csv("inputs.csv", header=headerList, index=False)
          
        df2 = pd.read_csv("inputs.csv")
        #self.train = train
        #self.shuffle = shuffle
        self.n_planes_for_train = n_planes_for_train
        self.gt_oplanes = df2.loc[:,"gt_oplane"]
        self.img_dir = df2.loc[:,"image"]
        self.depth_dir = df2.loc[:,"depth"]
        self.visible_dir = df2.loc[:,"visible"]
        #self.transform = transform
        #self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dir)

    def grey_transform(self, np_array):
        #print(np_array.shape)
        grey_image = cv2.cvtColor(np_array, cv2.COLOR_BGR2GRAY)
        return grey_image

    def concatenate_channels(self, img1, img2, axis=2):
        return np.concatenate((img1,img2), axis=2)
      
    def calculate_euclidean(self, np_array):
        return scipy.ndimage.distance_transform_edt(np_array)

    def calculate_edges(self, grey_img):
        edges = filters.farid(grey_img)
        return edges

    def resize_image(self, np_array, x_dim, y_dim):
        image_size = (x_dim, y_dim)
        image_resized = cv2.resize(np_array, image_size, interpolation=cv2.INTER_LINEAR)
        return image_resized

    def add_channels(self, np_array):
        img_resized = self.resize_image(np_array, 512, 512)

        grey_img = self.grey_transform(img_resized)
        edges = self.calculate_edges(grey_img)
        edges_resized = self.resize_image(edges, 512, 512)
        edges_resized = torch.from_numpy(edges_resized).unsqueeze(-1)
        edges_resized = edges_resized.cpu().detach().numpy()

        euc_dist = self.calculate_euclidean(grey_img)
        euc_dist_resized = torch.from_numpy(euc_dist).unsqueeze(-1)
        euc_dist_resized = euc_dist_resized.cpu().detach().numpy()

        concatenated_input = self.concatenate_channels(img_resized,euc_dist_resized, axis=2)
        concatenated_input = self.concatenate_channels(concatenated_input,edges_resized, axis=2)

        return concatenated_input

    def __getitem__(self, idx):
        #self.__len__
        #if self.shuffle:
        #  idx = random.randint(0,self.__len__)

        #if self.train:

        image = cv2.imread(self.img_dir[idx])
        image = self.add_channels(image)
        label_obj = MeshObj(self.gt_oplanes[idx],max_depth_range,mesh_data_root,binvoxPathPrefix,extra_mesh_data_root)
        label = label_obj.Get_GroundTruth(self.n_planes_for_train)

        depth = np.load(self.depth_dir[idx]).view()
        depth = self.resize_image(depth, 512,512)
        depth = np.expand_dims(depth, axis=0)

        image = np.moveaxis(image, -1, 0)
        label = np.moveaxis(label, -1, 0)

        return image, label, depth


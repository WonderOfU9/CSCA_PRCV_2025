"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
from __future__ import print_function
from datafuncs import ImageFolderWithRatings, ImageFolderWithRatingsAndFilenames_VARS, CustomDataLoader, get_subset, ImageFolderWithRatingsAndFilenames
from invert import Invert
import numpy as np
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch import nn, optim
import torchvision
from torchvision import transforms

class AuDrADataModule(pl.LightningDataModule):
    def __init__(self, args, data_dir = '/home/cbl/IQA/zihao/AuDrA_Drawings/primary_images', ratings_path="/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/primary_jrt.csv"):
        super().__init__()
        self.data_dir = data_dir
        self.ratings_path = ratings_path
        self.args = args
        self.transform = transforms.Compose([
            Invert(),
            transforms.Resize(self.args.in_shape[-1]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [self.args.img_means,self.args.img_means,self.args.img_means],
                std = [self.args.img_stds,self.args.img_stds,self.args.img_stds]
            )
        ])
        self.scaler = MinMaxScaler()

    def setup(self, stage = None):
        #  load and normalize ratings
        ratings = pd.read_csv("/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/primary_jrt.csv", header = None).to_numpy()
        self.scaler.fit(ratings [:,1].reshape((-1,1)))
        ratings[:,1] = self.scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))

        #  load images
        self.data = ImageFolderWithRatingsAndFilenames(
            root = self.data_dir,
            transform = self.transform,
            ratings = ratings
        )

        #  get indices
        train_count = int(len(self.data) * self.args.train_pct)
        val_count = int(len(self.data) * self.args.val_pct)
        indices = torch.randperm(len(self.data))

        data_indices = {"train" : get_subset(indices, 0, train_count),
                             "val" : get_subset(indices, train_count, val_count),
                             "test" : get_subset(indices, train_count + val_count, len(indices))
        }


        self.training_set = Subset(self.data, indices = data_indices["train"])
        self.validation_set = Subset(self.data, indices = data_indices["val"])
        self.test_set = Subset(self.data, indices = data_indices["test"])

        # return data_indices

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size = self.args.batch_size, drop_last = True, num_workers = 20)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size = self.args.batch_size, drop_last = True, num_workers = 20)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.args.batch_size,  num_workers = 20)
    def all_dataloaders(self):
        return DataLoader(self.data, batch_size=self.args.batch_size, num_workers=20)

def RaterGeneralizationOneDataloader(args, data_dir = '/home/cbl/IQA/zihao/AuDrA_Drawings/rater_generalization_one_images', ratings_path = '/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/rg1_jrt.csv'):
    transform = transforms.Compose([
        Invert(),
        transforms.Resize(args.in_shape[-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [args.img_means,args.img_means,args.img_means],
            std = [args.img_stds,args.img_stds,args.img_stds]
        )
    ])

    scaler = MinMaxScaler()
    ratings = pd.read_csv(args.rg1_ratings_path, header = None).to_numpy()
    scaler.fit(ratings [:,1].reshape((-1,1)))
    ratings[:,1] = scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))


    data = ImageFolderWithRatingsAndFilenames(
        root = args.rg1_data_dir,
        transform = transform,
        ratings = ratings
    )
    rg1_loader = DataLoader(data, batch_size = args.batch_size)

    return rg1_loader

def FarGeneralizationDataloader(args, data_dir = '/home/cbl/IQA/zihao/AuDrA_Drawings/far_generalization_images', ratings_path = 'fg_jrt.csv'):
    transform = transforms.Compose([
        Invert(),
        transforms.Resize(args.in_shape[-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [args.img_means,args.img_means,args.img_means],
            std = [args.img_stds,args.img_stds,args.img_stds]
        )
    ])

    scaler = MinMaxScaler()
    ratings = pd.read_csv(args.fg_ratings_path, header = None).to_numpy()
    scaler.fit(ratings [:,1].reshape((-1,1)))
    ratings[:,1] = scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))


    data = ImageFolderWithRatingsAndFilenames(
        root = args.fg_data_dir,
        transform = transform,
        ratings = ratings
    )
    fg_loader = DataLoader(data, batch_size = args.batch_size)

    return fg_loader

def RaterGeneralizationTwoDataloader(args, data_dir = '/home/cbl/IQA/zihao/AuDrA_Drawings/rater_generalization_two_images', ratings_path = 'rg2_jrt.csv'):
    transform = transforms.Compose([
        Invert(),
        transforms.Resize(args.in_shape[-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [args.img_means,args.img_means,args.img_means],
            std = [args.img_stds,args.img_stds,args.img_stds]
        )
    ])

    scaler = MinMaxScaler()
    ratings = pd.read_csv(args.rg2_ratings_path, header = None).to_numpy()
    scaler.fit(ratings [:,1].reshape((-1,1)))
    ratings[:,1] = scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))


    data = ImageFolderWithRatingsAndFilenames(
        root = args.rg2_data_dir,
        transform = transform,
        ratings = ratings
    )
    rg2_loader = DataLoader(data, batch_size = args.batch_size)

    return rg2_loader

def user_Dataloader(args, data_dir = 'user_images/'):
    transform = transforms.Compose([
        Invert(),
        transforms.Resize(args.in_shape[-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [args.img_means,args.img_means,args.img_means],
            std = [args.img_stds,args.img_stds,args.img_stds]
        )
    ])

    data = CustomDataLoader(
        main_dir = data_dir,
        transform = transform,
    )

    user_loader = DataLoader(data, batch_size = 1)

    return user_loader



class AuDrADataModule_VARS(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir =     args.training_data_dir
        self.ratings_path = args.ratings_path
        self.content_path = args.contents_path

        self.ratings = None
        self.contents = None

        self.transform = transforms.Compose([
            Invert(),
            transforms.Resize(self.args.in_shape[-1]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [self.args.img_means,self.args.img_means,self.args.img_means],
                std = [self.args.img_stds,self.args.img_stds,self.args.img_stds]
            )
        ])
        self.scaler = MinMaxScaler()

    def setup(self, stage = None):
        #  load and normalize ratings
        ratings = pd.read_csv(self.ratings_path, header = None).to_numpy()
        self.scaler.fit(ratings [:,1].reshape((-1,1)))
        ratings[:,1] = self.scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))
        self.ratings = ratings
        print(ratings.shape)



        contents = pd.read_csv(self.content_path,header = None).to_numpy()
        contents[:, 1] = contents[:, 1] - 1
        self.contents = contents
        print(contents.shape)

        #  load images
        self.data = ImageFolderWithRatingsAndFilenames_VARS(
            root = self.data_dir,
            transform = self.transform,
            ratings = ratings,
            contents = contents
            #styles = styles,
        )

        #  get indices
        train_count = int(len(self.data) * self.args.train_pct)
        val_count = int(len(self.data) * self.args.val_pct)
        indices = torch.randperm(len(self.data))

        data_indices = {"train" : get_subset(indices, 0, train_count),
                             "val" : get_subset(indices, train_count, val_count),
                             "test" : get_subset(indices, train_count + val_count, len(indices))
        }


        self.training_set = Subset(self.data, indices = data_indices["train"])
        self.validation_set = Subset(self.data, indices = data_indices["val"])
        self.test_set = Subset(self.data, indices = data_indices["test"])

        # return data_indices

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size = self.args.batch_size, drop_last = True, num_workers = 20,shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size = self.args.batch_size, drop_last = True, num_workers = 20)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.args.batch_size,  num_workers = 20)

    def all_dataloaders(self):
        return DataLoader(self.data, batch_size=self.args.batch_size, num_workers=20)

class AuDrADataModule_VARSS(pl.LightningDataModule):
    def __init__(self, args, data_dir = '/home/cbl/IQA/zihao/AuDrA_Drawings/primary_images', ratings_path="/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/primary_jrt.csv",style_path="/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/2.csv",content_path="/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/1.csv"):
        super().__init__()
        self.data_dir = data_dir
        self.ratings_path = ratings_path

        self.style_path = style_path    #风格文件的csv
        self.content_path = content_path#内容文件csv

        self.ratings = None
        self.styles = None
        self.contents = None

        self.args = args
        self.transform = transforms.Compose([
            Invert(),
            transforms.Resize(self.args.in_shape[-1]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [self.args.img_means,self.args.img_means,self.args.img_means],
                std = [self.args.img_stds,self.args.img_stds,self.args.img_stds]
            )
        ])
        self.scaler = MinMaxScaler()

    def setup(self, stage = None):
        #  load and normalize ratings
        ratings = pd.read_csv("/home/cbl/IQA/zihao/osfstorage-archive/Additional_Resources/AuDrA_Train/primary_jrt.csv", header = None).to_numpy()
        self.scaler.fit(ratings [:,1].reshape((-1,1)))
        ratings[:,1] = self.scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))
        self.ratings = ratings
        print(ratings.shape)

        styles = pd.read_csv(self.style_path,header = None).to_numpy()
        styles[:,2] = styles[:,2] - 1
        self.styles = styles
        print(styles.shape)




        contents = pd.read_csv(self.content_path,header = None).to_numpy()
        contents[:, 1] = contents[:, 1] - 1
        self.contents = contents
        print(contents.shape)

        #  load images
        self.data = ImageFolderWithRatingsAndFilenames_VARS(
            root = self.data_dir,
            transform = self.transform,
            ratings = ratings,
            contents = contents,
            styles = styles,
        )

        #  get indices
        #train_count = int(len(self.data) * self.args.train_pct)
        #val_count = int(len(self.data) * self.args.val_pct)
        #indices = torch.randperm(len(self.data))

        train_count = int(len(self.data) * (self.args.train_pct + self.args.val_pct))  # 合并训练集和验证集
        indices = torch.randperm(len(self.data))

        data_indices = {"train": get_subset(indices, 0, train_count),
                        "test": get_subset(indices, train_count, len(indices))}

        #data_indices = {"train" : get_subset(indices, 0, train_count),
        #                     "val" : get_subset(indices, train_count, val_count),
        #                     "test" : get_subset(indices, train_count + val_count, len(indices))
        #}


        #self.training_set = Subset(self.data, indices = data_indices["train"])
        #self.validation_set = Subset(self.data, indices = data_indices["val"])
        #self.test_set = Subset(self.data, indices = data_indices["test"])

        self.training_set = Subset(self.data, indices=data_indices["train"])
        self.test_set = Subset(self.data, indices=data_indices["test"])



        # return data_indices

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size = self.args.batch_size, drop_last = True, num_workers = 20,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.args.batch_size,  num_workers = 20)
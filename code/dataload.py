from torch.utils.data import Dataset
import os
import pandas as pd
from skimage import io, transform


class ASLTrainDataset(Dataset):
    """A custom DataLoader for out  pytorch model"""
    def __init__(self, Datasetfolder, transforms=None):

        os.chdir(Datasetfolder + '/Train')
        self.Datasetfolder = Datasetfolder
        self.img_name_lst = []
        self.data_dict = {}
        self.labels = os.listdir()
        self.transform = transform

        for label in self.labels:
            os.chdir(label)
            images = os.listdir()
            self.data_dict.update({label: images})
            os.chdir('..')
        imag_names = self.data_dict.values()
        for name in imag_names:
            for _ in name:
                self.img_name_lst.append(_)
        self.img_name_lst = pd.Series(self.img_name_lst)
        self.length = len(self.img_name_lst)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        try:
            os.chdir(self.Datasetfolder + '/Train')
        except FileNotFoundError:
            print("File Not Found")
        train_image = self.img_name_lst[index]
        os.chdir(train_image[0])
        image = io.imread(train_image)
        label = self.img_name_lst[index][0]
        os.chdir('..')
        sample = {'image': image, 'label': label}
        return sample


class ASLTestDataset(Dataset):
    def __init__(self, Datasetfolder):
        os.chdir(Datasetfolder + '/Test')
        self.Datasetfolder = Datasetfolder
        self.img_name_lst = []
        self.labels = os.listdir()
        self.transform = transform
        for label in self.labels:
            self.img_name_lst.append(label)
        self.length = len(self.img_name_lst)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        try:
            os.chdir(self.Datasetfolder + '/Test')
        except FileNotFoundError:
            print('File Not Found')
        image = io.imread(self.img_name_lst[index])
        label = self.img_name_lst[index][0]
        sample = {'image': image, 'label': label}
        return sample

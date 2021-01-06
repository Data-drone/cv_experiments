from pytorch_lightning import LightningDataModule

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import albumentations as A
from PIL import Image
import numpy as np

class AlbumentationTransform(object):
    def __call__(self, img):
        aug = A.Compose([
            A.Resize(200, 300),
            A.CenterCrop(100, 100),
            A.RandomCrop(80, 80),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-90, 90)),
            A.VerticalFlip(p=0.5)
        ])
        return Image.fromarray(aug(image=np.array(img))['image'])


class BasicPipe(LightningDataModule):
    def __init__(self, hparams, train_datadir, val_datadir, mean, std):
        super().__init__()

        self.hparams = hparams
        self.train_datadir=train_datadir
        self.val_datadir=val_datadir
        self.dataset_mean = mean
        self.dataset_std = std

        data_transform_normal = transforms.Compose([
            transforms.Resize((300,300)),
            transforms.CenterCrop((100, 100)),
            transforms.RandomCrop((80, 80)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
            
    
        data_transform = transforms.Compose([
                AlbumentationTransform(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    
        val_data_transform = transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        self.train_transforms = data_transform
        self.val_transforms = val_data_transform

    def setup(self, stage=None):
        
        self.train_data = ImageFolder(
            root=self.train_datadir,
            transform = self.train_transforms
        )
    
        self.val_data = ImageFolder(
            root=self.val_datadir,
            transform = self.val_transforms
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, 
                                num_workers=self.hparams.nworkers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, 
                                num_workers=self.hparams.nworkers)
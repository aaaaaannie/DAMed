import monai
from monai import transforms
import torch
from torch.utils.data import Dataset

class TrainData(Dataset):
    def __init__(self, cfg):
        super(TrainData, self).__init__()
        self.cfg      = cfg
        self.samples  = []
        with open(cfg.root+'/train.txt', 'r', encoding='utf-8') as lines:
            for line in lines:
                image, mask, age = line.strip().split(',')
                self.samples.append({'image':cfg.root+'/'+image, 'mask':cfg.root+'/'+mask, 'age':float(age)})
        ## transformation
        self.transform  = transforms.Compose([
                                transforms.LoadImaged(keys=['image', 'mask']),
                                transforms.EnsureChannelFirstd(keys=['image', 'mask']),
                                transforms.ScaleIntensityd(keys=['image', 'mask']),
                                transforms.Resized(keys=["image", "mask"], spatial_size=[480, 480]),
                                transforms.CenterSpatialCropd(keys=["image", "mask"], roi_size=(cfg.image_size, cfg.image_size)),
                                transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=(0,1)),
                                transforms.RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=1, spatial_axes=(0,1)),
                            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample          = self.samples[idx]
        sample          = self.transform(sample)
        return sample



class ValidData(Dataset):
    def __init__(self, cfg):
        super(ValidData, self).__init__()
        ## read attributes
        self.cfg      = cfg
        self.samples  = []
        with open(cfg.root+'/valid.txt', 'r', encoding='utf-8') as lines:
            for line in lines:
                image, mask, age = line.strip().split(',')
                self.samples.append({'image':cfg.root+'/'+image, 'mask':cfg.root+'/'+mask, 'age':float(age)})
        ## transform
        self.transform  = transforms.Compose([
                                transforms.LoadImaged(keys=['image', 'mask']),
                                transforms.EnsureChannelFirstd(keys=['image', 'mask']),
                                transforms.ScaleIntensityd(keys=['image', 'mask']),
                                transforms.Resized(keys=["image", "mask"], spatial_size=[480, 480]),
                                transforms.CenterSpatialCropd(keys=["image", "mask"], roi_size=(cfg.image_size, cfg.image_size)),
                            ])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample          = self.samples[idx]
        sample          = self.transform(sample)
        return sample



class TestData(Dataset):
    def __init__(self, cfg):
        super(TestData, self).__init__()
        ## read attributes
        self.cfg      = cfg
        self.samples  = []
        with open(cfg.root+'/test.txt', 'r', encoding='utf-8') as lines:
            for line in lines:
                image, mask, age = line.strip().split(',')
                self.samples.append({'image':cfg.root+'/'+image, 'mask':cfg.root+'/'+mask, 'age':float(age)})
        ## transform
        self.transform  = transforms.Compose([
                                transforms.LoadImaged(keys=['image', 'mask']),
                                transforms.EnsureChannelFirstd(keys=['image', 'mask']),
                                transforms.ScaleIntensityd(keys=['image', 'mask']),
                                transforms.Resized(keys=["image", "mask"], spatial_size=[480, 480]),
                                transforms.CenterSpatialCropd(keys=["image", "mask"], roi_size=(cfg.image_size, cfg.image_size)),
                            ])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample          = self.samples[idx]
        path            = sample['mask']
        sample          = self.transform(sample)
        return sample, path
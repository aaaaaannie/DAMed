import os
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import monai
from monai import transforms
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils import TestData

def test(cfg):
    ## dataset
    test_data                 = TestData(cfg)
    test_loader               = DataLoader(test_data, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
    #print("First sample type:", type(next(iter(test_loader))))
    ## model
    sam, img_embedding_size   = sam_model_registry['vit_b'](image_size=cfg.image_size, num_classes=1, checkpoint=cfg.snapshot, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    model                     = LoRA_Sam(sam, cfg.rank).cuda()
    model.load_lora_parameters(cfg.lora_snapshot)

    ## testing
    model.eval()
    with torch.no_grad():
        metric_attri=[[[],[]], [[],[]]]
        print('len(test_loader):',len(test_loader))
        for i, sample in enumerate(test_loader):
            #print(len(sample))
            #print(sample[0])
            # print('type of sample[image]:',type(sample['image'].cuda()))
            # print('sample[mask]:',sample['mask'])
            # print('sample[age]:',sample['age'])
            image, mask, age = sample[0]['image'].cuda(), sample[0]['mask'].cuda(), sample[0]['age'].item()
            mask        = (mask>0.5).float()
            pred        = model(image, False, cfg.image_size)['masks']
            pred        = (pred>0).float()
            inter       = (mask*pred).sum(dim=(1,2,3)).item()
            union       = (mask+pred).sum(dim=(1,2,3)).item()
            dice        = 2*inter/(union+1e-6)
            iou         = inter/(union-inter+1e-6)
            print(i, dice, iou)
            if age<60:
                metric_attri[0][0].append(dice)
                metric_attri[1][0].append(iou)
            else:
                metric_attri[0][1].append(dice)
                metric_attri[1][1].append(iou)
        
        dice_age_small60 = sum(metric_attri[0][0])/len(metric_attri[0][0])
        dice_age_large60 = sum(metric_attri[0][1])/len(metric_attri[0][1])
        dice_overall     = (sum(metric_attri[0][0])+sum(metric_attri[0][1]))/(len(metric_attri[0][0])+len(metric_attri[0][1]))
        dice_es          = dice_overall/(abs(dice_age_small60-dice_overall)+abs(dice_age_large60-dice_overall)+1)
        print('dice[es]=%.4f'%(dice_es))           
        print('dice[overall]=%.4f'%(dice_overall))           
        print('dice[age<60]=%.4f'%(dice_age_small60))            
        print('dice[age>=60]=%.4f'%(dice_age_large60))            

        iou_age_small60 = sum(metric_attri[1][0])/len(metric_attri[1][0])
        iou_age_large60 = sum(metric_attri[1][1])/len(metric_attri[1][1])
        iou_overall     = (sum(metric_attri[1][0])+sum(metric_attri[1][1]))/(len(metric_attri[1][0])+len(metric_attri[1][1]))
        iou_es          = iou_overall/(abs(iou_age_small60-iou_overall)+abs(iou_age_large60-iou_overall)+1)
        print('iou[es]=%.4f'%(iou_es))           
        print('iou[overall]=%.4f'%(iou_overall))           
        print('iou[age<60]=%.4f'%(iou_age_small60))            
        print('iou[age>=60]=%.4f'%(iou_age_large60))        


if __name__=='__main__':
    class Config:
        def __init__(self):
            self.batch_size        = 1
            self.rank              = 4
            self.patch_size        = 16
            self.root              = '../dataset'
            self.image_size        = 352
            self.snapshot          = './checkpoints/sam_vit_b_01ec64.pth'
            #self.lora_snapshot     = './save-vrex/seed_2/model-30'
            self.lora_snapshot     = './results/baseline/seed_9741/model-28'

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    test(Config())
import os
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from monai import transforms
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
from utils import TrainData, ValidData, TestData
import torch
import numpy as np
import random
import os
import csv
from datetime import datetime
import sys

def set_seed(seed_number):
    # position of setting seeds also matters
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.random.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_results_to_csv(file_path, results):
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["epoch", "dice_es", "dice_overall", "dice_small", "dice_large", "dice_delta"])
        writer.writerow(results)
        
def train(cfg, seed):
    ## dataset
    g = torch.Generator()
    g.manual_seed(seed)
    def seed_worker(worker_id):
        np.random.seed(seed)
        random.seed(seed)
    train_data                = TrainData(cfg)
    train_loader              = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=True)
    
    valid_data                = ValidData(cfg)
    valid_loader              = DataLoader(valid_data, batch_size=cfg.batch_size, num_workers=4, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    
    test_data                 = TestData(cfg)
    test_loader               = DataLoader(test_data, batch_size=cfg.batch_size, num_workers=0, worker_init_fn=seed_worker, generator=g, pin_memory=True)   
    
    ## model
    sam, img_embedding_size   = sam_model_registry['vit_b'](image_size=cfg.image_size, num_classes=1, checkpoint=cfg.snapshot, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    model                     = LoRA_Sam(sam, cfg.rank, seed=seed).cuda()
    # optimizer                 = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer                 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.base_lr, betas=(0.9, 0.999), weight_decay=0.0001)

    
    # Initialize CSV files to log results
    valid_csv_file_path = os.path.join(cfg.save_path, 'validation_results.csv')
    test_csv_file_path = os.path.join(cfg.save_path, 'test_results.csv')
    
    ## logger
    #logger.add(cfg.save_path+'/train.log', filter=lambda x: "[train]" in x['message']) 
    #logger.add(cfg.save_path+'/valid.log',  filter=lambda x: "[valid]" in x['message']) 

    for epoch in range(cfg.max_epochs):
        # # adjust lr
        # if epoch+1 in [12, 24, 36, 48]:
        #     optimizer.param_groups[0]['lr'] *= 0.5

        model.train()
        for i, sample in enumerate(train_loader):
            # # visualize
            # image, mask, age  = sample['image'], sample['mask'], sample['age']
            # print(image.shape, mask.shape, age)
            # print(torch.unique(image))
            # plt.subplot(221)
            # image = image[0,0].numpy()
            # plt.imshow(image)
            # mask  = mask[0,0].numpy()
            # mask  = (mask>0.5)
            # print(np.unique(mask))
            # plt.subplot(222)
            # plt.imshow(mask)
            # plt.subplot(223)
            # image = image*0.5+mask*0.5
            # plt.imshow(image)
            # plt.savefig('visualize.png')
            # input()

            image, mask = sample['image'].cuda(), sample['mask'].cuda()
            mask        = (mask>0.5).float()
            pred        = model(image, False, cfg.image_size)['masks']
            pred        = torch.sigmoid(pred)
            loss_ce     = -mask*torch.log(pred+1e-6)-(1-mask)*torch.log(1-pred+1e-6)
            q           = torch.exp(0.01*loss_ce.detach())
            loss_ce     = (loss_ce*q).sum()/q.sum()

            loss_dice   = 1-(2*mask*pred).sum(dim=(1,2,3))/((mask+pred).sum(dim=(1,2,3))+1e-6)
            q           = torch.exp(0.01*loss_dice.detach())
            loss_dice   = (loss_dice*q).sum()/q.sum()
            loss        = 0.2*loss_ce + 0.8*loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10==0:
                logger.info("[train][seed=%d] | iter=%d/%d | epoch=%d/%d | lr=%.4f | loss_ce=%.4f | loss_dice=%.4f"%(seed,i, len(train_loader), epoch+1, cfg.max_epochs, optimizer.param_groups[0]['lr'], loss_ce.item(), loss_dice.item()))

        ## validation
        model.eval()
        with torch.no_grad():
            dice_overall, cnt_overall, dice_small, cnt_small, dice_large, cnt_large = 0, 0, 0, 0, 0, 0
            for i, sample in enumerate(valid_loader):
                image, mask, age = sample['image'].cuda(), sample['mask'].cuda(), sample['age']
                mask             = (mask>0.5).float()
                pred             = model(image, False, cfg.image_size)['masks']
                pred             = (pred>0).float()
                dice             = 2*(mask*pred).sum(dim=(1,2,3))/((mask+pred).sum(dim=(1,2,3))+1e-6)

                dice_overall    += sum(dice)
                cnt_overall     += len(dice)
                dice_small      += sum(dice[age<60])
                cnt_small       += len(dice[age<60])
                dice_large      += sum(dice[age>=60])
                cnt_large       += len(dice[age>=60])

            dice_overall /= cnt_overall
            dice_small   /= cnt_small
            dice_large   /= cnt_large
            dice_es       = dice_overall/(abs(dice_small-dice_overall)+abs(dice_large-dice_overall)+1)  
            
            logger.info("[valid][seed=%d] | epoch=%d | dice[overall]=%.4f | dice[age<60]=%.4f | dice[age>=60]=%.4f | dice[delta]=%.4f"%(seed,epoch+1, dice_overall, dice_small, dice_large, abs(dice_small-dice_large)))

            # Save validation results to CSV
            save_results_to_csv(valid_csv_file_path, [epoch + 1, dice_es.item(), dice_overall.item(), dice_small.item(), dice_large.item(), abs(dice_small - dice_large).item()])
            
        ## save model
        model.save_lora_parameters(cfg.save_path+'/model-'+str(epoch+1))
   
        ## test
        model_path = os.path.join(cfg.save_path+'/model-'+str(epoch+1))
        model.load_lora_parameters(model_path)
        model.eval()
        with torch.no_grad():
            dice_overall, cnt_overall, dice_small, cnt_small, dice_large, cnt_large = 0, 0, 0, 0, 0, 0
            for i, sample in enumerate(test_loader):
                image, mask, age = sample[0]['image'].cuda(), sample[0]['mask'].cuda(), sample[0]['age']
                mask             = (mask>0.5).float()
                pred             = model(image, False, cfg.image_size)['masks']
                pred             = (pred>0).float()
                dice             = 2*(mask*pred).sum(dim=(1,2,3))/((mask+pred).sum(dim=(1,2,3))+1e-6)

                dice_overall    += sum(dice)
                cnt_overall     += len(dice)
                dice_small      += sum(dice[age<60])
                cnt_small       += len(dice[age<60])
                dice_large      += sum(dice[age>=60])
                cnt_large       += len(dice[age>=60])


            dice_overall /= cnt_overall
            dice_small   /= cnt_small
            dice_large   /= cnt_large
            dice_es       = dice_overall/(abs(dice_small-dice_overall)+abs(dice_large-dice_overall)+1)
    
            logger.info("[test][seed=%d] | epoch=%d | dice[overall]=%.4f | dice[age<60]=%.4f | dice[age>=60]=%.4f | dice[delta]=%.4f"%(seed,epoch+1, dice_overall, dice_small, dice_large, abs(dice_small-dice_large)))
             
            # Save test results to CSV
            save_results_to_csv(test_csv_file_path, [epoch + 1, dice_es.item(), dice_overall.item(), dice_small.item(), dice_large.item(), abs(dice_small - dice_large).item()])
        
        

if __name__=='__main__':
    seed = int(sys.argv[1])
    gpu = sys.argv[2]
        
    class Config:
        def __init__(self,seed):
            self.batch_size        = 32
            self.max_epochs        = 30
            self.base_lr           = 0.0001
            self.rank              = 4
            self.patch_size        = 16
            self.root              = '../dataset'
            self.image_size        = 352
            now = datetime.now()
            self.local_time = now.strftime("%m%d-%H%M%S")
            self.save_path         = f'./results/groupDRO/seed_{seed}/{self.local_time}'
            self.snapshot          = './checkpoints/sam_vit_b_01ec64.pth'
            os.makedirs(self.save_path, exist_ok=True)
    
    os.environ['CUDA_VISIBLE_DEVICES']=gpu

    logger.remove()
    set_seed(seed)
    print(f"Running training with seed: {seed}")
    cfg = Config(seed)
    logger.add(os.path.join(cfg.save_path, 'train.log'), filter=lambda x: "[train]" in x['message'])
    logger.add(os.path.join(cfg.save_path, 'valid.log'), filter=lambda x: "[valid]" in x['message'])
    train(cfg, seed)
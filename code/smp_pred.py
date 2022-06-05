import cv2
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data
import numpy as np
import skimage
import pandas as pd
import glob
import matplotlib.pyplot as plt

import argparse
from numpy import  asarray
from numpy import  ones_like

import segmentation_models_pytorch as smp

from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler

from albumentations.pytorch import ToTensorV2
import albumentations as A
def predict():
    test_transform = A.Compose([
            # A.Resize(480,480),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std= [0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.)
    all_models = ['Unetscse_b3']
    temp_dir = 'E:\PycharmProjects\hualu\model'
    models = []
    for m in all_models:
        for i in range(1):
            if m == 'unetpp_b3_ns':
                model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b3', encoder_weights=None, classes=2)
                loading_dir = os.path.join(temp_dir,m+'_fold_' + str(i) + '.pth')
                print('loading %s from %s'%(m,loading_dir))
                model.load_state_dict(torch.load(loading_dir))
            elif m == 'FPN_b3_ns':
                model = smp.FPN(encoder_name='timm-efficientnet-b3', encoder_weights=None, classes=2)
                loading_dir = os.path.join(temp_dir,m+'_fold_' + str(i) + '.pth')
                print('loading %s from %s'%(m,loading_dir))
                model.load_state_dict(torch.load(loading_dir))
            elif m == 'Unetscse_b3':
                model = smp.Unet(encoder_name='timm-efficientnet-b3', encoder_weights=None, decoder_attention_type='scse',
                                classes=2)
                loading_dir = os.path.join(temp_dir,m+'_fold_' + str(i) + '.pth')
                print('loading %s from %s'%(m,loading_dir))
                model.load_state_dict(torch.load(loading_dir))          
            models.append(model)

    if not os.path.exists('result//result'): # this is not true
        os.makedirs('result//result')
    rot = torchvision.transforms.functional.rotate
    def sum_8(model, tensor):
        batch = torch.cat([tensor, tensor.flip([-1]), tensor.flip([-2]), tensor.flip([-1,-2]), rot(tensor,90).flip([-1]), rot(tensor,90).flip([-2]), rot(tensor,90).flip([-1,-2]), rot(tensor,90)], 0)
        pred = model(batch).detach()
        return pred[:1]+pred[1:2].flip([-1])+pred[2:3].flip([-2])+pred[3:4].flip([-1,-2])+rot(pred[4:5].flip([-1]),-90)+rot(pred[5:6].flip([-2]),-90)+rot(pred[6:7].flip([-1,-2]),-90)+rot(pred[7:],-90)
    def add_8_logi(models, tensor):
        results=torch.zeros(1,2,480,480,device='cuda')
        for model in models:
            model.to('cuda').eval()
            results += model(tensor.to('cuda'))
        return results
    all_test_imgs = glob.glob('E:\\PycharmProjects\\hualu\\raw_data\\round_train\\train_fusai\\train_org_image\\*.png')[1:2]
    print( all_test_imgs)
    names=[]
    count=0
    for i in all_test_imgs:
        count+=1
        print(count)
        file_name = i.split('\\')[-1]
        print(file_name)
        names.append(file_name)
        img = cv2.imread(i)[:,:,::-1]
        size = img.shape[:2]
        tensor = torch.unsqueeze(test_transform(image=img)['image'],0)
        # mask = torch.squeeze(add_8_logi(models, tensor).argmax(1)).to('cpu').numpy()
        mask = models[0](tensor)
        # mask = torch.squeeze(add_8_logi(models, tensor).argmax(1)).to('cpu').numpy()        # cv2.imwrite(os.path.join('result',file_name),cv2.resize((mask*255), size[::-1], interpolation=cv2.INTER_NEAREST))
# predict()
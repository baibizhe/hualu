import cv2
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils import data
import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt

import argparse
from numpy import  asarray
from numpy import  ones_like

import timm

import segmentation_models_pytorch as smp

from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler

from albumentations.pytorch import ToTensorV2
import albumentations as A
def class_pred():
    test_transform = A.Compose([
            A.Resize(640,640),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std= [0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.)
    rot = torchvision.transforms.functional.rotate
    def esb_8(model, tensor):
        batch = torch.cat([tensor, tensor.flip([-1]), tensor.flip([-2]), tensor.flip([-1,-2]), rot(tensor,90).flip([-1]), rot(tensor,90).flip([-2]), rot(tensor,90).flip([-1,-2]), rot(tensor,90)], 0)
        pred = model(batch).detach()
        return pred[0]+pred[1]+pred[2]+pred[3]+pred[4]+pred[5]+pred[6]+pred[7]
    def Esb(models, tensor):
        results=torch.zeros([4], device='cuda')
        for model in models:
            model.to('cuda').eval()
            results+=esb_8(model, tensor.to('cuda'))
        return np.argmax(results.cpu().numpy())
    models = []
    for i in range(5):
        model = torchvision.models.mobilenet_v3_large(pretrained=False, num_classes=4)
        models_dir = '/home/project/model/'
        model.load_state_dict(torch.load(models_dir+'classfi_mbnet'+'_fold_' + str(i)+'.pth'))
        model.classifier[3] = nn.Linear(in_features=1280, out_features=4, bias=True)
        models.append(model)
    all_test_imgs = glob.glob('/home/project/raw_data/round_test/*.png')[1:100]
    names=[]
    labels=[]
    count=0
    for i in all_test_imgs:
        count+=1
        print(count)
        file_name = i.split('/')[-1]
        names.append(file_name)
        img = cv2.imread(i)[:,:,::-1]
        tensor = torch.unsqueeze(test_transform(image=img)['image'],0)
        labels.append(Esb(models, tensor))
    df = pd.DataFrame()
    df['image_name'] = names
    df['label'] = labels
    df.to_csv('/home/project/temp_data/result/result.csv',index=False)
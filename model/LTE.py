import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import MeanShift
import cv2
import numpy as np
import GPUtil

#Learning Texture Extractor
class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()
        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features
    
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x]) # slice 1 contains VGG layer 0 - 1
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x]) # slice 2 contains VGG layer 2 - 6
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x]) # slice 3 contains VGG layer 7 - 11
        
            
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std) #defined in utils.py: Object! No single Value.

    def forward(self, x):
        #print("vorher")
        #GPUtil.showUtilization()# 39%
        #y = x*2*127.5
        #cv2.imshow('y', np.transpose(y.detach().squeeze().round().cpu().numpy(),(1,2,0)).astype(np.uint8))
        #cv2.waitKey(0)
        x = self.sub_mean(x) #defined above in self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)- ???? x is image matrix 3x160x160 --- Shift RGB Mean values in a way VGG works best (as in imagenet)
        #y = x*2*127.5
        #cv2.imshow('y', np.transpose(y.detach().squeeze().round().cpu().numpy(),(1,2,0)).astype(np.uint8))
        #cv2.waitKey(0)
        #GPUtil.showUtilization() 39%
        x = self.slice1(x) #output not viewable       
        #GPUtil.showUtilization() 45%
        x_lv1 = x #shape = 1,64,160,160       
        #GPUtil.showUtilization() 45%
        x = self.slice2(x)
        x_lv2 = x 
        x = self.slice3(x) #output of VGG after 2 (64x64) ,7 (128x128) and 12(256x256) Layers
        x_lv3 = x
        #print("nachher")
        #GPUtil.showUtilization()
        return x_lv1, x_lv2, x_lv3
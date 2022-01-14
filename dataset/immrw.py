import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random
import re
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()  # numpy.rot90(m, k=1, axes=(0, 1))---- Rotate an array by 90 degree k times -- In this case k = random number between 0 and 3
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        #sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        #sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1): #np.random.randint(0,2) can be 0 or 1.... Some Objects flipped others not
            sample['LR'] = np.fliplr(sample['LR']).copy()  #np.fliplr -- Flip array in the left/right direction.
            sample['HR'] = np.fliplr(sample['HR']).copy()
            #sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        #if (np.random.randint(0, 2) == 1):
            #sample['Ref'] = np.fliplr(sample['Ref']).copy()
            #sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            #sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        #if (np.random.randint(0, 2) == 1):
         #   sample['Ref'] = np.flipud(sample['Ref']).copy()
         #   sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample

class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR = sample['LR'], sample['LR_sr'], sample['HR']
        LR = LR.transpose((2,0,1)) #
        LR_sr = LR_sr.transpose((2,0,1))        
        HR = HR.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float()}
                

class ToTensorBox(object): #Only used in debug mode
    def __call__(self, sample):
        LR, LR_sr, HR, Box, srcFile = sample['LR'], sample['LR_sr'], sample['HR'], sample['Box'], sample['srcFile']
        LR = LR.transpose((2,0,1)) #
        LR_sr = LR_sr.transpose((2,0,1))        
        HR = HR.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Box': Box,
                'srcFile': srcFile}

class ToTensorRef(object): #Only used for ref images
    def __call__(self, sample):
        Ref, Ref_sr = sample['Ref'], sample['Ref_sr']
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}
        
  
class RandomCrop(object):
    def __init__(self,args):
        self.args = args
    
    def __call__(self, sample):
        h,w = sample.shape[:2]
        hpx=self.args.ref_crop_size
        wpx=self.args.ref_crop_size
        hStart = torch.randint(0, h-hpx,(1,))
        wStart = torch.randint(0, w-wpx,(1,))
        crop = sample[hStart:hStart+hpx,wStart:wStart+wpx]
        return crop                


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.LR_list = sorted([os.path.join(args.dataset_dir, 'train/LR/', name) for name in #added / to end of path to make it windows compatible
            os.listdir( os.path.join(args.dataset_dir, 'train/LR/') )]) # lists names of files in directory
        self.HR_list = sorted([os.path.join(args.dataset_dir, 'train/HR/', name) for name in #added / to end of path to make it windows compatible
            os.listdir( os.path.join(args.dataset_dir, 'train/HR/') )]) # lists names of files in directory
        self.transform = transform
        self.args = args
    def __len__(self):
        if len(self.LR_list) == len(self.HR_list):
            out = len(self.LR_list)
        else:
            out = "HR and LR not of equal length - Error!"
        return out
        

    def __getitem__(self, idx):
        ### HR       
        HR = imread(self.HR_list[idx])
        LR = imread(self.LR_list[idx])
        
        HR = np.array([HR, HR, HR]).transpose(1,2,0) #make RGB image from greyscale imput----- Solve differently later!
        LR = np.array([LR, LR, LR]).transpose(1,2,0)
        #HR = np.array(HRpre).transpose(1,2,0)
        #LR = np.array(LRpre).transpose(1,2,0)
        
        #HR = imread(self.input_list[idx])
        #HR = np.array(Image.fromarray(HR).resize((160, 160), Image.BICUBIC))
        h,w = HR.shape[:2]

        #HR = HR[:h//4*4, :w//4*4, :]
        ### LR and LR_sr
        #LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC)) #HR image to LR image via bicubic... to 1/4 of height and width   --- Image.fromarray makes PILLOW Image from original
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC)) # Resize LR back to HR with bicubic

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR}

        if self.transform:
            sample = self.transform(sample)
        return sample

class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        self.HR_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'valid/HR/', '*.tif'))) #path had to be altered manually --- Why *_0.png files for test?
        self.LR_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'valid/LR/', '*.tif')))  
        self.args = args
        if self.args.debug:
            self.dfTiles = pd.read_csv(os.path.join(args.dataset_dir, 'tiles.csv'))
            self.box = []
            self.srcFile = []
            for LR in self.LR_list:
                box = self.dfTiles.loc[self.dfTiles['LRfilename'] == os.path.basename(LR)].box.item() 
                box = [int(s) for s in re.findall(r'-?\d+\.?\d*', box)]
                self.box.append(box)
                self.srcFile.append(LR[-6:])
            del box
            del self.dfTiles
            self.transform = transforms.Compose([ToTensorBox()])
        else:
            self.transform = transform

     
     
    def __len__(self):
        if len(self.LR_list) == len(self.HR_list):
            out = len(self.LR_list)
        else:
            out = "HR and LR not of equal length - Error!"
        return out

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.HR_list[idx])
        LR = imread(self.LR_list[idx])
        if self.args.debug:
            box = self.box[idx]
            srcFile = self.srcFile[idx]
            
        HR = np.array([HR, HR, HR]).transpose(1,2,0) #make RGB image from greyscale imput----- Solve differently later!
        LR = np.array([LR, LR, LR]).transpose(1,2,0)
        
        
        h, w = HR.shape[:2]
        h, w = h//4*4, w//4*4
        HR = HR[:h, :w, :] ### crop to the multiple of 4

        ### LR and LR_sr
        #LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))
        
        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        
        if self.args.debug:
            sample = {'LR': LR,  
                    'LR_sr': LR_sr,
                    'HR': HR,
                    'Box': box,
                    'srcFile':srcFile}
        else:
            sample = {'LR': LR,  
                    'LR_sr': LR_sr,
                    'HR': HR}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
class RefSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensorRef()]) ):
        self.ref_list = sorted([os.path.join(args.reference_dir, name) for name in #added / to end of path to make it windows compatible
            os.listdir( os.path.join(args.reference_dir))])  #added / to end of path to make it windows compatible
        self.transform = transform
        self.args = args
        if self.args.ref_crop_size:
            self.croper = RandomCrop(self.args)
    
    def __len__(self):
        if len(self.ref_list) == self.args.NumbRef:
            out = len(self.LR_list)
        elif len(self.ref_list) > self.args.NumbRef:
            out = "Too many files in reference image folder. Expecting "+str(self.args.NumbRef)+" files. Changes to the reference images make a new model training necessary. Change --NumbRef in options.py therefore!"
        else:
            out = "Not enough files in reference image folder. Expecting "+str(self.args.NumbRef)+" files. Changes to the reference images make a new model training necessary. Change --NumbRef in options.py therefore!"    
        return out
        
    def __getitem__(self, idx):       
        ### Ref and Ref_sr
        Ref_sub = imread(self.ref_list[idx]) 
        #Ref_sub.show()
        if self.args.ref_crop_size:
            Ref_sub = self.croper(Ref_sub) #Take a Random 160 x 160 Crop
            
        if self.args.gray:
            Ref_sub = np.array([Ref_sub, Ref_sub, Ref_sub]).transpose(1,2,0) #make RGB image from greyscale imput
               
        h2, w2 = Ref_sub.shape[:2]
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))
    
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        if self.args.ref_crop_size:
            Ref = np.zeros((self.args.ref_crop_size, self.args.ref_crop_size, 3))
            Ref_sr = np.zeros((self.args.ref_crop_size, self.args.ref_crop_size, 3))
        elif not self.args.ref_image_size:
            print("Either ref_crop_size or ref_image_size has to be set! Both are = None! See options.py")
        else:
            Ref = np.zeros((self.args.ref_image_size, self.args.ref_image_size, 3))
            Ref_sr = np.zeros((self.args.ref_image_size, self.args.ref_image_size, 3))
            
        Ref[:h2, :w2] = Ref_sub
        Ref_sr[:h2, :w2] = Ref_sr_sub

        ### change type
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample    
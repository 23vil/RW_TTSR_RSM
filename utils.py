import math
import numpy as np
import logging
import cv2
import os
import shutil
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


class plotter:
    def __init__(self, losstype, ylabel):
        self.data = np.empty(0)
        self.epoch = np.empty(0)
        self.batch = np.empty(0)
        self.losstype = losstype
        self.ylabel = ylabel
    def store(self, data):
        self.data = np.append(self.data,data[2])
        self.batch = np.append(self.batch,data[1])
        self.epoch = np.append(self.epoch,data[0])
    def write(self, directory): 
        # Data for plotting
        font = {'weight' : 'normal',
        'size'   : 11.5}

        plt.rc('font', **font)
        fig, ax = plt.subplots(figsize=(20,5))
        ax.plot(self.batch+max(self.batch)*self.epoch, self.data)

        ax.set(xlabel='Epoch', ylabel=self.ylabel,
               title=self.losstype)
        ax.set_xticks(np.unique(self.epoch)*(len(self.batch)/len(np.unique(self.epoch))))

        ax.set_xticklabels(np.unique(self.epoch))
        ax.grid()

        fig.savefig(os.path.join(directory,str(self.losstype)+'.png'))
        #plt.show()
    def debug(self):
        try:
            print(self.data)
            print(self.epoch)
        except:
            print("no t or s")
        print(self.data)


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        if (not args.reset):
            raise SystemExit('Error: save_dir "' + args.save_dir + '" already exists! Please set --reset True to delete the folder.')
        else:
            shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir)
    # os.makedirs(os.path.join(args.save_dir, 'img'))

    if ((not args.eval) and (not args.test)):
        os.makedirs(os.path.join(args.save_dir, 'model'))
    
    if ((args.eval and args.eval_save_results) or args.test):
        os.makedirs(os.path.join(args.save_dir, 'save_results'))

    args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name), 
        logger_name=args.logger_name).get_log()

    return _logger


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1) #sets up a nn.Conv2d with 3 in_channels, 3 out_channels and kernel size 1
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) #torch.eye(3) creates 3x3 matrix with 1 in diagonal an 0 elsewhere...... .view() reshapes the tensor. In this case to three tensors of three 1x1 tensors
        self.weight.data.div_(std.view(3, 1, 1, 1)) #divide self.weight.data through std ... Inverts std to 1/std
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) #negative RGB_mean
        self.bias.data.div_(std) #-RGB_mean/std = -"coefficient of variation" also called "relative std".... for VGG19 standard data
        self.weight.requires_grad = False
        self.bias.requires_grad = False


def calc_psnr(img1, img2):
    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0    #
    diff[:,:,0] = diff[:,:,0] * 65.738 / 256.0  #blue  Maybe normalization of colour range? Transformation to YCBCR colorspace
    diff[:,:,1] = diff[:,:,1] * 129.057 / 256.0 #green
    diff[:,:,2] = diff[:,:,2] * 25.064 / 256.0 #red
    diff = np.sum(diff, axis=2) #keeps 3 different sums for RGB
    mse = np.mean(np.power(diff, 2)) # mean of squared error- term per color
        #mse = np.mean( (img1 - img2) ** 2 )
        #if mse == 0:
        #    return 100
        #PIXEL_MAX = 255.0 # Maxmimum Pixel Value is 255 
    return -10 * math.log10(mse) # negative (-10) as MSE/PIXEL_MAX is inverted and supposed to be PIXEL_MAX/MSE
    #return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
  
def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2 # both default for SSIM 

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calc_psnr_and_ssim(sr, hr):
    ### args:
        # sr: pytorch tensor, range [-1, 1]
        # hr: pytorch tensor, range [-1, 1]

    ### prepare data
    sr = (sr+1.) * 127.5 #reverse transform from [-1,1]-space to [0,255]-rgb
    hr = (hr+1.) * 127.5
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]
    img1 = np.transpose(sr.squeeze().round().cpu().numpy(), (1,2,0))
    img2 = np.transpose(hr.squeeze().round().cpu().numpy(), (1,2,0))
    psnr = calc_psnr(img1, img2)
    ssim = calc_ssim(img1, img2)

    return psnr, ssim
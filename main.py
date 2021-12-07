from option import args
from utils import mkExpDir

from dataset import dataloader
#from dataset import RefRelevance

from model import TTSR
from model import RefSelector
from loss.loss import get_loss_dict
from trainer import Trainer
from torch.nn.parallel import DistributedDataParallel
import numpy as np

import torch.nn as nn
import os
import torch
import warnings
from sys import exit
warnings.filterwarnings('ignore')
 

if __name__ == '__main__':
    ### Create Save Directory and start training progress logger
    _logger = mkExpDir(args)
    
    
    ### Initialize dataloader of training dataset and testing dataset
    _dataloader = dataloader.get_dataloader(args) 
    
    
    ### Initialize device and TTSR model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = TTSR.TTSR(args, device)#.to(device)
    _model = _model.to(device)
    
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu))).to(device)       
    
    
    ### Initialize loss calculation
    _loss_all = get_loss_dict(args, _logger, device) #defined in loss/loss.pt
    
    
    ### trainer.py initialization --> In this file the training, evaluation happens. Also test images are created in here.
    t = Trainer(args, device,  _logger, _dataloader, _model, _loss_all)
    
    
    
    ### Start the porgramm (e.g. test / eval / train) = execute one of the blocks below   
    ##Test Mode - load pretrained model --> then test
    if (args.test):      
        t.load(model_path=args.model_path)
        t.test()
    
    
    ##Evaluation Mode - load pretrained model --> then evaluate    
    elif (args.eval):   
        t.load(model_path=args.model_path)
        t.evaluate()
    
    
    ##Load pretrained Model and continue training this model   
    elif (args.retrain):        
        t.load(model_path=args.model_path)
        #t.evaluate()
        for epoch in range(1, args.num_init_epochs+1):  #Initialization epoch
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1): #Full Training Epochs
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
     
     
    ##Train new model from scratch            
    else:       
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
            t.TotLossPlotter.write(args.save_dir, is_init=True)
            t.RecLossPlotter.write(args.save_dir, is_init=True)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            t.TotLossPlotter.write(args.save_dir)
            t.AdvLossPlotter.write(args.save_dir)
            t.PerLossPlotter.write(args.save_dir)
            t.RecLossPlotter.write(args.save_dir)
            t.TplLossPlotter.write(args.save_dir)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)

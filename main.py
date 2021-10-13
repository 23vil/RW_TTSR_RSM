from option import args
from utils import mkExpDir
from dataset import dataloader
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
warnings.filterwarnings('ignore')

def free_device(args): #Check for free device and return respective device, also considering the option.py configuration (cpu/gpu).
    if args.num_gpu ==1 :
        def get_freer_gpu():
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            return np.argmax(memory_available)
        return torch.device('cpu' if args.cpu else 'cuda:'+str(get_freer_gpu()))
    else:
        return torch.device('cpu' if args.cpu else 'cuda')    

if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) #if (not args.test) else None

    ### device and model
    device = free_device(args)
    _model = TTSR.TTSR(args).to(device)
    if args.seperateRefLoss:
        _RefSelModel = RefSelector.RefSelector(args).to(device)
        
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))       
        #_model = DistributedDataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger) #defined in loss/loss.pt
    
    ### trainer initialization
    if args.seperateRefLoss: ## Train with seperated RSM
        t = Trainer(args, device, _logger, _dataloader, _RefSelModel, _model, _loss_all)
    else: ##Train with RSM included in main model
        t = Trainer(args, device,  _logger, _dataloader, None, _model, _loss_all)

    ### test / eval / train
    if (args.test):      ##Test Mode
        t.load(model_path=args.model_path)
        if args.seperateRefLoss:
            t.loadRef(ref_model_path=args.ref_model_path)
        t.test()
    elif (args.eval):   ##Evaluation Mode
        t.load(model_path=args.model_path)
        if args.seperateRefLoss:
            t.loadRef(ref_model_path=args.ref_model_path)
        t.evaluate()
    elif (args.refTrain):       ##Train only Reference Selection Model
        for epoch in range(1, args.num_epochs+1):
            if (args.retrain):
                t.loadRef(ref_model_path=args.ref_model_path)
            t.refTrain()
    elif (args.retrain):        ##Load pretrained Model and continue training this Model
        t.load(model_path=args.model_path)
        if args.seperateRefLoss:
            t.loadRef(model_path=args.ref_model_path)
        t.evaluate()
        for epoch in range(1, args.num_init_epochs+1):  #Initialization epoch
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1): #Full Training Epochs
            t.train(current_epoch=epoch, is_init=False)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)
    else:       ##Train new model from scratch
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            t.TotLossPlotter.write(args.save_dir)
            t.AdvLossPlotter.write(args.save_dir)
            t.PerLossPlotter.write(args.save_dir)
            t.RecLossPlotter.write(args.save_dir)
            t.TplLossPlotter.write(args.save_dir)
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)

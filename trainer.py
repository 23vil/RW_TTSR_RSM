from utils import calc_psnr_and_ssim
from utils import plotter
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import re

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
from torch.nn.parallel import DistributedDataParallel

import GPUtil

import time


class Trainer():
    def __init__(self, args, device, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model            
        self.loss_all = loss_all
        self.device = device
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        self.updateRefVectorDict()
            
        #variables for saving best models
        self.bestLoss = None
        self.nameOfmodel = None
        self.nameOfmodel_ssim = None
        self.nameOfmodel_psnr = None
        self.best_psnr = None
        self.best_ssim = None        
        self.RecLossPlotter = plotter(losstype="Training_Rec_Loss", ylabel="Loss")
        self.PerLossPlotter = plotter(losstype="Training_Per_Loss", ylabel="Loss")
        self.TplLossPlotter = plotter(losstype="Training_Tpl_Loss", ylabel="Loss")
        self.AdvLossPlotter = plotter(losstype="Training_Adv_Loss", ylabel="Loss")
        self.TotLossPlotter = plotter(losstype="Training_Total_Loss", ylabel="Loss")
        
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))
        self.BatchNumber = len(self.dataloader['train'])
        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if 
            args.num_gpu==1 else self.model.module.MainNet.parameters()),
            "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if 
            args.num_gpu==1 else self.model.module.LTE.parameters()), 
            "lr": args.lr_rate_lte
            }
        ]
        if not self.args.eval and not self.args.test:
            self.optimizer = optim.AdamW(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
            #self.scheduler = optim.lr_scheduler.StepLR(
                #self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.args.lr_base, max_lr=self.args.lr_max, cycle_momentum=False, step_size_up=int(round(2*self.BatchNumber/3)), step_size_down=self.BatchNumber-int(round(2*self.BatchNumber/3)))

        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0
        
        if (self.args.gray_transform):
            self.transformGray = torchvision.transforms.Grayscale(num_output_channels=3)
    
    def updateRefVectorDict(self):
        with torch.no_grad():
            for RefID in range(self.args.NumbRef):
                Refsr = self.dataloader['ref'][RefID]['Ref_sr'].unsqueeze(dim=0).to(self.device)
                self.model.RefSelector.updateRefVectorDict(Refsr, RefID, self.model.LTE)
            
    def load(self, model_path=None, discr_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)
        if (discr_path):
            self.loss_all['adv_loss'].update_state_dict(discr_path)
                
            
    def loadOptimAndDiscriminator(self, optim_path=None, discr_path=None, discr_optim_path=None):
        if (optim_path and discr_path and discr_optim_path):
            self.logger.info('load_optim_path: ' + optim_path)
            optim_state_dict_save = {k:v for k,v in torch.load(optim_path, map_location=self.device).items()}
            optim_state_dict = self.optimizer.state_dict()
            optim_state_dict.update(optim_state_dict_save)
            self.optimizer.load_state_dict(optim_state_dict)
            
            self.logger.info('load_discr_path: ' + discr_path)
            self.logger.info('load_discr_optim_path: ' + discr_path)
            self.loss_all['adv_loss'].update_state_dict(discr_path, discr_optim_path)
            
            
            
    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched
    
    
    
    
    
### def train defines the training of a model -------------------------------------------------------------------------------------------------
   
    def train(self, current_epoch=0, is_init=False):
        
        self.model.train() #self.model is TTSR.py from model folder
        
        #self.scheduler.step()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))
        
            
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()
            
            ##Prepare Batch
            sample_batched = self.prepare(sample_batched) 
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            with torch.no_grad():
                refID = self.model.RefSelector(lr_sr)

            reftmp = []
            ref_srtmp = []
            for ID in refID:
                reftmp.append(self.dataloader['ref'][ID]['Ref']) 
                ref_srtmp.append(self.dataloader['ref'][ID]['Ref_sr'])
            ref = torch.stack((reftmp),0).to(self.device)
            ref_sr = torch.stack((ref_srtmp),0).to(self.device)
            
            ##Run Super Resolution
            sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)#, ref=ref, refsr=ref_sr) #Here the models output for the inputs (arguments) is requested. (forward pass)
            
            ##Transfrom RGB SR output to greyscale
            if (self.args.gray_transform):
                sr = self.transformGray(sr)
            
            ### Calc and Store losses
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print # only print if the batch index is dividable by "print_every"
                
            #Reconstruction Loss
            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr) #rec_w = weight of reconstruction loss - defined in train.sh (also default value in option.py) ---- loss_all defined in loss/loss.pt by "get_loss_dict" - change variable name in main.py
            loss = rec_loss
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )
                self.RecLossPlotter.store([current_epoch-1,i_batch, rec_loss.item()])
                
            #Perceptual Loss    
            if (not is_init):
                if ('per_loss' in self.loss_all):
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                    per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                    loss += per_loss
                    if (is_print):
                        self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
                        self.PerLossPlotter.store([current_epoch-1,i_batch, per_loss.item()])
                        
            #Transferal Perceptual Loss
                if ('tpl_loss' in self.loss_all):
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr) # bezieht sr_lv1, sr_lv2, sr_lv3 aus LTE_copy
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, 
                        S, T_lv3, T_lv2, T_lv1)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )
                        self.TplLossPlotter.store([current_epoch-1,i_batch, tpl_loss.item()])
                        
            #Adversarial Loss
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr,learningrate=self.optimizer.param_groups[0]['lr'])
                    loss += adv_loss
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )  
                        self.AdvLossPlotter.store([current_epoch-1,i_batch, adv_loss.item()])
            #Total Loss
            if (is_print):    
                self.logger.info( 'total_loss: %.10f' %(loss.item()) )
                self.TotLossPlotter.store([current_epoch-1,i_batch, loss.item()])
            
            #Backpropagate
            loss.backward()           
            self.optimizer.step()
            self.scheduler.step()
            self.updateRefVectorDict()
            
        ##Save Model & Discriminator
        if ((not is_init) and current_epoch % self.args.save_every == 0):
            #TTSR Model
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)
            del tmp
            
            if self.args.SaveOptimAndDiscr:
                self.logger.info('saving the optimizer...')
                tmp = self.optimizer.state_dict()
                optimizer_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
                model_name = self.args.save_dir.strip('/')+'/model/optimizer_'+str(current_epoch).zfill(5)+'.pt'
                torch.save(optimizer_state_dict, model_name)
                
                
                #Discriminator
                self.logger.info('saving the discriminator...')
                tmp_model, tmp_optim = self.loss_all['adv_loss'].get_state_dict()
                
                discriminator_state_dict = {key.replace('module.',''): tmp_model[key] for key in tmp_model if 
                    (('SearchNet' not in key) and ('_copy' not in key))}
                model_name = self.args.save_dir.strip('/')+'/model/discriminator_'+str(current_epoch).zfill(5)+'.pt'
                torch.save(discriminator_state_dict, model_name)
                
                self.logger.info('saving the discriminator optimizer...')
                discr_optim_state_dict = {key.replace('module.',''): tmp_optim[key] for key in tmp_optim if 
                    (('SearchNet' not in key) and ('_copy' not in key))}
                model_name = self.args.save_dir.strip('/')+'/model/discriminator_optim_'+str(current_epoch).zfill(5)+'.pt'
                torch.save(discr_optim_state_dict, model_name)
            
        
        if ((is_init) and current_epoch % self.args.save_every == 0):
            #TTSR Model in Initial Trailing
            self.logger.info('saving the init model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/init-model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)
            del tmp
          
         
         
         
         
### Evaluation is defined here----------------------------------------------------------------------------------------         
         
    def evaluate(self, current_epoch=0, is_init=False):
        
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
        if self.args.debug:
            if not is_init:
                self.evalLossesDf = pd.DataFrame(columns=['Box','srcFile','TotLoss','RecLoss','PerLoss','TplLoss','AdvLoss'])
            else:
                self.evalLossesDf = pd.DataFrame(columns=['Box','srcFile','TotLoss','RecLoss'])
        if (self.args.dataset):# == 'IMM'):  #Dataset name has to be manually altered here.. better Solution!
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt, rec_loss, per_loss, tpl_loss, adv_loss = 0., 0., 0, 0., 0., 0., 0.
                for i_batch, sample_batched in enumerate(self.dataloader['test']):
                    cnt += 1
                    if (self.args.debug):                      
                        srcFile = sample_batched['srcFile']
                        box = sample_batched['Box']
                        del sample_batched['Box']
                        del sample_batched['srcFile']
                        srcFile = srcFile[0]
                        for i, boxItem in enumerate(box):
                            box[i] = boxItem.item()                        
                    sample_batched = self.prepare(sample_batched)
                    
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    
                    ### Refernce Selection
                    refID = self.model.RefSelector(lr)                
                    reftmp = []
                    ref_srtmp = []
                    for ID in refID:
                        reftmp.append(self.dataloader['ref'][ID]['Ref']) 
                        ref_srtmp.append(self.dataloader['ref'][ID]['Ref_sr'])
                    ref = torch.stack((reftmp),0).to(self.device)
                    ref_sr = torch.stack((ref_srtmp),0).to(self.device)
                    del refID
                    del reftmp
                    del ref_srtmp
                    
                    ###Compute SR
                    sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    
                    ###Transform SR to gray
                    if (self.args.gray_transform):
                        sr = self.transformGray(sr)
                    
                    #print(adv_loss.element_size()*adv_loss.nelement())
                    
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8) # squeeze delets all 1 values --- round() rounds to closest integer -- .cpu() moves object to cpu ---.numpy transforms object to numpy array--- transform to np.unit8 type
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.tif'), sr_save)
                        
                    ### calculate psnr and ssim
                    
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    psnr += _psnr #dtype float
                    ssim += _ssim #dtype float
                    
                    ###calc loss
                    ##Rec Loss
                    rec_loss_tmp = self.args.rec_w * self.loss_all['rec_loss'](sr, hr) #rec_w = weight of reconstruction loss - defined in train.sh (also default value in option.py) ---- loss_all defined in loss/loss.pt by "get_loss_dict" - change variable name in main.py
                    rec_loss += rec_loss_tmp #both stay at 4 byte
                    rec_loss_tmp = float(rec_loss_tmp)
                    
                    
                    if not is_init:
                        
                        ##Per.Loss
                        sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                        with torch.no_grad():
                            hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                            
                        per_loss_tmp = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                        per_loss += per_loss_tmp
                        per_loss_tmp =float(per_loss_tmp)
                        
                        ##Tpl Loss
                        sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr, no_backward = True)
                        
                        tpl_loss_tmp = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, S, T_lv3, T_lv2, T_lv1)
                        tpl_loss += tpl_loss_tmp
                        tpl_loss_tmp = float(tpl_loss_tmp)
                        
                        ##Adv Loss
                        #with torch.enable_grad():
                        adv_loss_tmp = self.args.adv_w * self.loss_all['adv_loss'](sr, hr, no_backward = True)
                        adv_loss += adv_loss_tmp #both stay at 4 bytes
                        adv_loss_tmp = float(adv_loss_tmp)
                        
                        #Store Losses for Loss-HeatMap generation
                        if self.args.debug:
                            self.evalLossesDf = self.evalLossesDf.append({
                                                                        'Box':box ,
                                                                        'srcFile':srcFile,
                                                                        'TotLoss':rec_loss_tmp + per_loss_tmp + tpl_loss_tmp + adv_loss_tmp,
                                                                        'RecLoss':rec_loss_tmp,
                                                                        'PerLoss':per_loss_tmp,
                                                                        'TplLoss':tpl_loss_tmp,
                                                                        'AdvLoss':adv_loss_tmp},ignore_index=True)
                    else:
                        if self.args.debug:
                            self.evalLossesDf = self.evalLossesDf.append({
                                                                        'Box':box ,
                                                                        'srcFile':srcFile,
                                                                        'TotLoss':rec_loss_tmp.cpu(),
                                                                        'RecLoss':rec_loss_tmp.cpu()},ignore_index=True)
                        
                ###Save Loss HeatMaps
                if self.args.debug:
                    originalImages = np.unique(self.evalLossesDf.srcFile.values)
                    heatMaps = {}
                    heatCounter = {}
                    #for i, orig in enumerate(originalImages):
                    #    originalImages[i] = orig[0]
                    for orig in originalImages:
                        heatMaps[orig] = np.zeros((5,1024,1024)) #LR Image resolution
                        heatCounter[orig] = np.zeros((5,1024,1024),dtype=int)
                    for datum in self.evalLossesDf.itertuples():
                        box = datum.Box
                        #box = [int(s) for s in re.findall(r'-?\d+\.?\d*', box)]
                        for line in range(box[0],box[2]):
                            for row in range(box[1],box[3]):
                                heatMaps[datum.srcFile][0][line][row]+= datum.TotLoss
                                heatCounter[datum.srcFile][0][line][row]+= 1
                                heatMaps[datum.srcFile][1][line][row]+= datum.RecLoss
                                heatCounter[datum.srcFile][1][line][row]+= 1
                                if not is_init:
                                    heatMaps[datum.srcFile][2][line][row]+= datum.PerLoss
                                    heatCounter[datum.srcFile][2][line][row]+= 1
                                    heatMaps[datum.srcFile][3][line][row]+= datum.TplLoss
                                    heatCounter[datum.srcFile][3][line][row]+= 1
                                    heatMaps[datum.srcFile][4][line][row]+= datum.AdvLoss
                                    heatCounter[datum.srcFile][4][line][row]+= 1
                                
                    
                    if not is_init:
                        for image in originalImages:                       
                            for i, lossName in enumerate(['TotLoss', 'RecLoss', 'PerLoss', 'TplLoss', 'AdvLoss']):
                                heatMaps[image][i] = np.divide(heatMaps[image][i],heatCounter[image][i])
                                fig, ax = plt.subplots(figsize=(10,10))
                                sns.set(rc={'axes.facecolor':'gray', 'figure.facecolor':'white',"font.size":11,"axes.titlesize":11,"axes.labelsize":11})
                                im = plt.imread(os.path.join("/home/ps815691/datasources/fixed/IMMresVar1024-4096/lr/valid/",image))
                                ax = sns.heatmap(heatMaps[image][i], linewidth=0, linecolor='black', cmap='jet', xticklabels = 200, yticklabels = 200, cbar_kws={"shrink": 0.75})#, vmin= 23)
                                ax = ax.imshow(im, extent=[0, 1024, 0, 1024])
                                plt.savefig(os.path.join(self.args.save_dir,str(current_epoch+1)+"_"+image[-6:-4]+"_"+lossName+'.png'))
                                #plt.close(fig)
                    else:
                        for image in originalImages:
                            for i, lossName in enumerate(['TotLoss', 'RecLoss']):
                                heatMaps[image][i] = np.divide(heatMaps[image][i],heatCounter[image][i])
                                fig, ax = plt.subplots(figsize=(10,10))
                                sns.set(rc={'axes.facecolor':'gray', 'figure.facecolor':'white',"font.size":11,"axes.titlesize":11,"axes.labelsize":11})
                                im = plt.imread(os.path.join("/home/ps815691/datasources/fixed/IMMresVar1024-4096/lr/valid/",image))
                                ax = sns.heatmap(heatMaps[image][i], linewidth=0, linecolor='black', cmap='jet', xticklabels = 200, yticklabels = 200, cbar_kws={"shrink": 0.75})#, vmin= 23)
                                ax = ax.imshow(im, extent=[0, 1024, 0, 1024])
                                plt.savefig(os.path.join(self.args.save_dir,str(current_epoch+1)+"_init_"+image[-6:-4]+"_"+lossName+'.png'))
                                #plt.close(fig)
                    if is_init:
                        self.evalLossesDf.to_csv(os.path.join(self.args.save_dir,str(current_epoch)+"_init-epoch_lossheatMap.csv"))
                    else:
                        self.evalLossesDf.to_csv(os.path.join(self.args.save_dir,str(current_epoch)+"_epoch_lossheatMap.csv"))    
                        

                        
                ###average & print losses and calc total val_loss
                #print(rec_loss) sometimes NAN
                rec_loss = rec_loss / cnt 
                if not is_init:
                    per_loss = per_loss / cnt
                    tpl_loss = tpl_loss / cnt
                    adv_loss = adv_loss / cnt
                    val_loss = rec_loss + per_loss + tpl_loss + adv_loss 
                else: val_loss = rec_loss
                
              
                self.logger.info( 'val_rec_loss: %.10f' %(rec_loss.item()) )
                if not is_init:
                    self.logger.info( 'val_per_loss: %.10f' %(per_loss.item()) )
                    self.logger.info( 'val_tpl_loss: %.10f' %(tpl_loss.item()) )
                    self.logger.info( 'val_adv_loss: %.10f' %(adv_loss.item()) )
                self.logger.info( 'total_val_loss: %.10f' %(val_loss.item()) )
                
                ###save best model according to loss
                if self.bestLoss:
                    if self.bestLoss > val_loss:
                        self.bestLoss = val_loss 
                        self.logger.info('saving the model...')
                        tmp = self.model.state_dict()
                        model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                            (('SearchNet' not in key) and ('_copy' not in key))}
                        if self.nameOfmodel:
                            os.remove(str(self.nameOfmodel))
                        self.nameOfmodel = self.args.save_dir.strip('/')+'/model/best_model_loss_'+str(current_epoch)+'.pt'
                        torch.save(model_state_dict, self.nameOfmodel)
                else:
                    self.bestLoss = val_loss
                
                ###save best model according to psnr
                psnr_ave = psnr / cnt
                if self.best_psnr:
                    if self.best_psnr < psnr_ave:
                        self.best_psnr = psnr_ave                         
                        self.logger.info('saving the model...')
                        tmp = self.model.state_dict()
                        model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                            (('SearchNet' not in key) and ('_copy' not in key))}
                        if self.nameOfmodel_psnr:
                            os.remove(str(self.nameOfmodel_psnr))
                        self.nameOfmodel_psnr = self.args.save_dir.strip('/')+'/model/best_model_psnr_'+str(current_epoch)+'.pt'
                        torch.save(model_state_dict, self.nameOfmodel_psnr)
                else:
                    self.best_psnr = psnr_ave
                
                
                ###save best model according to ssim
                ssim_ave = ssim / cnt
                if self.best_ssim:
                    if self.best_ssim < ssim_ave:
                        self.best_ssim = ssim_ave                         
                        self.logger.info('saving the model...')
                        tmp = self.model.state_dict()
                        model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                            (('SearchNet' not in key) and ('_copy' not in key))}
                        if self.nameOfmodel_ssim:
                            os.remove(str(self.nameOfmodel_ssim))
                        self.nameOfmodel_ssim = self.args.save_dir.strip('/')+'/model/best_model_ssim_'+str(current_epoch)+'.pt'
                        torch.save(model_state_dict, self.nameOfmodel_ssim)
                else:
                    self.best_ssim = ssim_ave
                
                
                
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')
        
####Generation of test images is defined below----------------------------------------------------------------------------
        
    def test(self):
        self.logger.info('Test process...')
        
        
        self.model.eval()
        
        ### LR and LR_sr
        if self.args.lr_path:
            self.logger.info('lr path:     %s' %(self.args.lr_path))
            LR = imread(self.args.lr_path)
 
            if self.args.gray:
                LR = np.array([LR, LR, LR]).transpose(1,2,0) #transformation to RGB
            h1, w1 = LR.shape[:2]
            LR_sr = np.array(Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))        
            
            ### change type
            LR = LR.astype(np.float32)
            LR_sr = LR_sr.astype(np.float32)

            ### rgb range to [-1, 1]
            LR = LR / 127.5 - 1.
            LR_sr = LR_sr / 127.5 - 1.

            ### to tensor
            LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
            LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

            ### Ref and Ref_sr
            refID = self.model.RefSelector(LR_t)        
            reftmp = []
            ref_srtmp = []
            for ID in refID:
                reftmp.append(self.dataloader['ref'][ID]['Ref']) 
                ref_srtmp.append(self.dataloader['ref'][ID]['Ref_sr'])
            Ref = torch.stack((reftmp),0).to(self.device)
            Ref_sr = torch.stack((ref_srtmp),0).to(self.device)


            
            with torch.no_grad():
                sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref, refsr=Ref_sr)
                if (self.args.gray_transform):
                    sr = self.transformGray(sr)
                sr_save = (sr+1.) * 127.5
                sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
                imsave(save_path, sr_save)
                self.logger.info('output path: %s' %(save_path))
        elif self.args.lr_folder:
            imageFiles = [os.path.join(self.args.lr_folder, f) for f in os.listdir(self.args.lr_folder) if f.endswith('.tif')]
            for lrPath in imageFiles:
                self.logger.info('lr path:     %s' %(lrPath))
                LR = imread(lrPath)
    
                if self.args.gray:
                    LR = np.array([LR, LR, LR]).transpose(1,2,0) #transformation to RGB
                h1, w1 = LR.shape[:2]
                LR_sr = np.array(Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))        
                
                ### change type
                LR = LR.astype(np.float32)
                LR_sr = LR_sr.astype(np.float32)

                ### rgb range to [-1, 1]
                LR = LR / 127.5 - 1.
                LR_sr = LR_sr / 127.5 - 1.

                ### to tensor
                LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
                LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

                ### Ref and Ref_sr
                refID = self.model.RefSelector(LR_t)        
                reftmp = []
                ref_srtmp = []
                for ID in refID:
                    reftmp.append(self.dataloader['ref'][ID]['Ref']) 
                    ref_srtmp.append(self.dataloader['ref'][ID]['Ref_sr'])
                Ref = torch.stack((reftmp),0).to(self.device)
                Ref_sr = torch.stack((ref_srtmp),0).to(self.device)


                
                with torch.no_grad():
                    sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref, refsr=Ref_sr)
                    if (self.args.gray_transform):
                        sr = self.transformGray(sr)
                    sr_save = (sr+1.) * 127.5
                    sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                    save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(lrPath))
                    imsave(save_path, sr_save)
                    self.logger.info('output path: %s' %(save_path))


        self.logger.info('Test over.')
        

        
        
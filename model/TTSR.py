from model import MainNet, LTE, SearchTransfer, RefSelector

import torch
import torch.nn as nn
import torch.nn.functional as F
import GPUtil



class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        #Create all objects for Model from different .py-files
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) ) #defined in options.py
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, 
            res_scale=args.res_scale)#all parameters defined in options.py - creates MainNet Object from model/MainNet.py
        self.LTE      = LTE.LTE(requires_grad=True) # create LTE object from model/LTE.py
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss #second LTE object
        self.SearchTransfer = SearchTransfer.SearchTransfer() #object from model/SearchTransfer.py
        #self.RefSelector = RefSelector.RefSelector(len(listdir(self.args.reference_dir)))


    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None, no_backward=False):
        #lr [N,RGB/gray,H,W]
        #lrsr [N,RGB/gray,H*scale,W*scale]
        #ref &refsr [N,RGB/gray, Hcrop, Wcrop]
        
        #GPUtil.showUtilization() 10%
        #if sr == "test":
            #print("LTE "+str(torch.cuda.current_device())+"    "+str(self.LTE.state_dict().keys()))
            #print("LTE_Copy "+str(torch.cuda.current_device())+"    "+str(self.LTE_copy.state_dict().keys()))
            #return None
            
        if (type(sr) != type(None)): #if sr is not None - but is None by default
            ### used in transferal perceptual loss
            #print("LTE "+str(torch.cuda.current_device())+"    "+str(self.LTE.state_dict().keys()))
            #print("LTE_Copy "+str(torch.cuda.current_device())+"    "+str(self.LTE_copy.state_dict().keys()))
            if not no_backward:
                self.LTE_copy.load_state_dict(self.LTE.state_dict())#, strict=False) # State_dict is part of nn.module - state_dictReturns a dictionary containing a whole state of the module./ I set strict to False, to enable parallel computing
                #copies state_dict of LTE to LTE_copy
                sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            else:
                sr_lv1, sr_lv2, sr_lv3 = self.LTE((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3
        if self.args.refTrain: #Calculate perceptual RSM - Loss during RSM-Pretraining
            with torch.no_grad():
                _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)       
                _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)
                ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)
                _, _, _, _, RelevanceTensor = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)                
            return RelevanceTensor
        
        ###if ref == None:
        ###    return self.RefSelector(lr)
        
        _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.) #.detach() is similar to no_grad - lrsr is argument of forward - arguments are defined in trainer.py lines 69 - 75       
        #GPUtil.showUtilization() 10%
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)
        #GPUtil.showUtilization() #38% geprüft
        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)
        #GPUtil.showUtilization()  67%
           #LTE results for ref, lrsr and refsr
        S, T_lv3, T_lv2, T_lv1, _ = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        #GPUtil.showUtilization() out of memory
        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1

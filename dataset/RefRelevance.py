from os.path import isfile, join, isdir
from os import listdir
import torch
class RefRelevanceMap():
    def __init__(self, args, model, dataloader, device):
        self.args = args
        self.refdir = self.args.reference_dir
        self.lrdir = self.args.dataset_dir
        self.dataloader = dataloader
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched
    
    def generateRelevanceMap(self):
        refTifs = [join(self.refdir, f) for f in listdir(self.refdir) if f.endswith('.tif')]
        
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            sample_batched = self.prepare(sample_batched) 
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            for i_ref, refSrc in enumerate(refTifs):
                ref = self.dataloader['ref'][i_ref]['Ref']
                ref_sr = self.dataloader['ref'][i_ref]['Ref_sr']
                RefRelevanceTensor = self.model(lr=None, lrsr=lr_sr, ref=ref, refsr=ref_sr, relevance=True)
        return(RefRelevanceTensor)
        
        
        #ref_srtmp = []
        #for ID in refID:
        #    reftmp.append(self.dataloader['ref'][ID]['Ref']) 
        #    ref_srtmp.append(self.dataloader['ref'][ID]['Ref_sr'])
        #Ref = torch.stack((reftmp),0).to(self.device)
        #Ref_sr = torch.stack((ref_srtmp),0).to(self.device)
class RefRelevanceMap():
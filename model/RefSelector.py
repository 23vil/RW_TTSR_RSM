import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import operator

class RefSelector(nn.Module):
    def __init__(self, args):
        super(RefSelector, self).__init__()
        self.args = args
        self.noRef = args.NumbRef
        self.RSel = models.resnet18(pretrained=True)
        for param in self.RSel.parameters():
            param.requires_grad_(False)# = True
        self.RSel.fc = nn.Linear(in_features=512, out_features=self.noRef, bias=True)
        #self.RSel.requires_grad_(True)
        #self.dataloader = dataloader
        #self.device = device
    
    def forward(self, LR):
        #if self.args.test or self.args.eval:
        #   self.RSel.eval()
        #refID  = torch.argmax(self.RSel(LR), dim=1)
       
        #reftmp = []
        #ref_srtmp = []
        #for ID in refID:
        #    reftmp.append(self.dataloader['ref'][ID]['Ref']) 
        #    ref_srtmp.append(self.dataloader['ref'][ID]['Ref_sr'])
        #ref = torch.stack((reftmp),0).to(self.device)
        #refsr = torch.stack((ref_srtmp),0).to(self.device)
        return self.RSel(LR)
    
class RefSelector2(nn.Module):
    def __init__(self, args, device):
        super(RefSelector2, self).__init__()
        self.args = args
        self.device = device
        self.model = models.vgg19(pretrained=True)
        self.Img2Vec = Img2Vec(device=self.device, model=self.model)
        #self.noRef = args.NumbRef
        #self.RSel = models.resnet18(pretrained=True)
        self.RefVectorDict = {}
        self.LTE = []
        
        for i in range (13):
            self.model.features[i] = nn.Identity()
        self.model.eval()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        
    def updateRefVectorDict(self, Ref, RefID, LTE):
        self.LTE = LTE
        LTEref = self.LTE(Ref)
        self.RefVectorDict[RefID] = self.Img2Vec.get_vec(LTEref)  
        
    def forward(self, LR, LTE):
        self.LTE = LTE
        LTElr = self.LTE(LR)
        LRVector = self.Img2Vec.get_vec(LTElr)
        CosSim = {}
        for RefID,RefVector in self.RefVectorDict.items():
            CosSim[RefID] = self.cos(RefVector, LRVector)
        BestRefID = max(CosSim, key=CosSim.get)
        
        return BestRefID   

class Img2Vec():
    def __init__(self, device, model, layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = device
        self.layer_output_size = layer_output_size
        self.model = model
        self.extraction_layer = self._get_model_and_layer(layer)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.scaler = transforms.Resize((150, 150))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=True):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            my_embedding = torch.zeros(len(img), self.layer_output_size)
            def copy_data(m, i, o):
                my_embedding.copy_(o.data)
            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)
            h.remove()
            if tensor:
                return my_embedding
            else:
                return my_embedding.numpy()[:, :]

        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
            my_embedding = torch.zeros(1, self.layer_output_size)
            def copy_data(m, i, o):
                my_embedding.copy_(o.data)
            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(image)
            h.remove()
            if tensor:
                return my_embedding
            else:
                return my_embedding.numpy()[0, :]

    def _get_model_and_layer(self, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        
        # VGG-19l
        
        if layer == 'default':
            layer = self.model.classifier[-4]
            self.layer_output_size = self.model.classifier[-4].in_features # should be 4096
        else:
                
            layer = self.model.classifier[-layer]

        return layer
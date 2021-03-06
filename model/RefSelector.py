import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
    
class RefSelector(nn.Module):
    def __init__(self, args, device):
        super(RefSelector, self).__init__()
        self.args = args
        self.device = device
        self.LTEmodel = models.vgg19(pretrained=True)
        self.Img2Vec = Img2Vec(LTEmodel=self.LTEmodel)
        self.RefVectorDict = {}
        self.LTE = []       
        for i in range (13):
            self.LTEmodel.features[i] = nn.Identity()
        self.LTEmodel.eval()
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        
    def updateRefVectorDict(self, RefSr, RefID, LTE):
        with torch.no_grad():
            self.LTE = LTE
            _ ,_ , LTEref = self.LTE(RefSr)
            self.RefVectorDict[RefID] = self.Img2Vec.get_vec(LTEref)
        
    def forward(self, LR):
        with torch.no_grad():
            _, _, LTElr = self.LTE(LR)
        LRVector = self.Img2Vec.get_vec(LTElr)
           
        MaxSimIDs = []
        for sampleID in range(len(LRVector)):
            sample = LRVector[sampleID]
            MaximumSimilarity = [0,0]
            for key, value in self.RefVectorDict.items():
                TempSimilarity = self.cos(value,sample)
                if TempSimilarity.item() > MaximumSimilarity[1]:
                    MaximumSimilarity[0]=key
                    MaximumSimilarity[1]=TempSimilarity.item()
            MaxSimIDs.append(MaximumSimilarity[0])
        return MaxSimIDs   

class Img2Vec():
    def __init__(self, LTEmodel, layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.layer_output_size = layer_output_size
        self.LTEmodel = LTEmodel
        self.extraction_layer = self._get_model_and_layer(layer)
        
        self.LTEmodel.eval()

    def get_vec(self, img, tensor=True):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if len(img) != 1:
            my_embedding = torch.zeros(len(img), self.layer_output_size)
            def copy_data(m, i, o):
                my_embedding.copy_(o.data)
            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.LTEmodel(img)
            h.remove()
            return my_embedding

        else:
            my_embedding = torch.zeros(1, self.layer_output_size)
            def copy_data(m, i, o):
                my_embedding.copy_(o.data)
            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.LTEmodel(img)
            h.remove()
            return my_embedding

    def _get_model_and_layer(self, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """       
        # VGG-19
        
        if layer == 'default':
            layer = self.LTEmodel.classifier[-4]
            self.layer_output_size = self.LTEmodel.classifier[-4].in_features # should be 4096
        else:               
            layer = self.LTEmodel.classifier[-layer]
        return layer
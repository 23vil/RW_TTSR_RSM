import torch
import torch.nn as nn
import torchvision.models as models

class RefSelector(nn.Module):
    def _init_(self, args):
        super(refSelector, self).__init__()
        self.args = args
        self.noRef = args.NumberRef
        self.RSel = models.resnet18(pretrained=True)
        for param in self.RSel.parameters():
            param.requires_grad = False
        resnet18.fc = nn.Linear(in_features=512, out_features=noRef, bias=True)
    
    def forward(self, LR):
        refID  = self.RSel(LR)
        
        return refID
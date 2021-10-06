import torch
import torch.nn as nn
import torchvision.models as models

class RefSelector(nn.Module):
    def __init__(self, args):
        super(RefSelector, self).__init__()
        self.args = args
        self.noRef = args.NumbRef
        self.RSel = models.resnet18(pretrained=True)
        for param in self.RSel.parameters():
            param.requires_grad = False
        self.RSel.fc = nn.Linear(in_features=512, out_features=self.noRef, bias=True)
    
    def forward(self, LR):
        refID  = torch.argmax(self.RSel(LR), dim=1)
        return refID
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        
        self.f.pop(-1) # gets rid of last layer (AdaptiveAvgPool2d(output_size=(1, 1)))
        self.f.append(nn.AdaptiveAvgPool2d(output_size=(2, 1))) # and replace with this
        self.f = nn.Sequential(*self.f)

        # projection head
        self.g1 = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        # projection head
        self.g2 = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))       

    def forward(self, x):
        x = self.f(x) # this is the ([1,2048, 2, 1]) feature vector
        x1 = x[:,:,0,:]
        x2 = x[:,:,1,:]

        feature1 = torch.flatten(x1, start_dim=1)
        feature2 = torch.flatten(x2, start_dim=1)
        out1 = self.g1(feature1)
        out2 = self.g2(feature2)
        return F.normalize(feature1, dim=-1), F.normalize(feature2, dim=-1), F.normalize(out1, dim=-1), F.normalize(out2, dim=-1)


from torchvision import models, transforms
from torchvision.models import ResNet101_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
 
class ResNet101(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet101, self).__init__(*args, **kwargs)
        resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, img):
        outs = []
        x = img 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        outs.append(x)  #4
        x = self.layer2(x)
        outs.append(x)  #8
        x = self.layer3(x)
        outs.append(x)  #16
        x = self.layer4(x)
        outs.append(x)  #32
        return outs
# decoding module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(F.adaptive_max_pool2d(x, output_size=(1, 1))))
        x = x * c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        x = x * s
        return x

class CBAM_Decoder(nn.Module):
    def __init__(self, feature_channels, embedding_dim):
        super().__init__()

        self.conv1 = ConvRelu(feature_channels[-1], embedding_dim, 1, 1, 0)
        self.blend1 = ConvRelu(embedding_dim, embedding_dim, 3, 1, 1)
        self.cbam1 = CBAM(embedding_dim)
        self.conv2 = ConvRelu(feature_channels[-2], embedding_dim, 1, 1, 0)
        self.blend2 = ConvRelu(embedding_dim*2, embedding_dim, 3, 1, 1)
        self.cbam2 = CBAM(embedding_dim)
        self.conv3 = ConvRelu(feature_channels[-3], embedding_dim, 1, 1, 0)
        self.blend3 = ConvRelu(embedding_dim*2, embedding_dim, 3, 1, 1)
        self.cbam3 = CBAM(embedding_dim)
        self.conv4 = ConvRelu(feature_channels[-4], embedding_dim, 1, 1, 0)
        self.blend4 = ConvRelu(embedding_dim*2, embedding_dim, 3, 1, 1)
        self.cbam4 = CBAM(embedding_dim)
        self.predictor = Conv(embedding_dim, 1, 3, 1, 1)

    def forward(self, app_mo_feats, return_raw=False):   
        x = self.conv1(app_mo_feats[-1])
        x = self.cbam1(self.blend1(x))
        s16 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv2(app_mo_feats[-2]), s16], dim=1)
        x = self.cbam2(self.blend2(x))
        s8 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv3(app_mo_feats[-3]), s8], dim=1)
        x = self.cbam3(self.blend3(x))
        s4 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = torch.cat([self.conv4(app_mo_feats[-4]), s4], dim=1)
        x = self.predictor(self.cbam4(self.blend4(x)))
        score = F.interpolate(x, scale_factor=4, mode='bicubic')
        if return_raw:
            return [score], x
        return [score]

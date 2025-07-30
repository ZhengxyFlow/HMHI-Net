import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Diceloss(nn.Module):
    def __init__(self, smooth=1e-5, activation='sigmoid'):
        super(Diceloss, self).__init__()
        
        if activation is None or activation == "none":
            self.activation_fn = lambda x: x
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            self.activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
        
        self.smooth= smooth
        
    def forward(self, pred, gt):
        r""" computational formula：
            dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
        """
        pred = self.activation_fn(pred)
    
        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
    
        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        dice_coe = (2 * intersection + self.smooth) / (unionset + self.smooth)
        
        return 1-dice_coe.sum() / N


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred, label):
        loss_bce = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = loss_bce* modulating_factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        # print(loss.mean(0).sum())
        return loss.mean() + 0.5*loss_bce.mean()

class IOUloss(nn.Module):
    def __init__(self,weight_background=1.0, weight_foreground=1.0, epsilon=1e-7):
        super(IOUloss, self).__init__()
        self.weight_background= weight_background
        self.weight_foreground =weight_foreground
    
    def forward(self, y_pred, y_label):
        B,_,_,_ = y_pred.shape
        intersection = (y_label * y_pred).view(B, -1)
        intersection = torch.sum(intersection, dim=-1)
        
        union=(y_label + y_pred - y_label * y_pred).view(B, -1)
        union = torch.sum(union, dim=-1)

        iou = (intersection + self.epsilon) / (union + self.epsilon)

        # Calculate the weighted IoU loss
        weighted_loss = -math.log(iou) * (self.weight_background * (1 - y_label) + self.weight_foreground * y_label)

        return weighted_loss.mean()
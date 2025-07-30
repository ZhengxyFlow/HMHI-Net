import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from typing import Optional

from .utils.memory_encoder import *
from .utils.utils import *
from .utils.transformer_modules import *
from .utils.position_encoding import *

class ObjectAttentionLayer(nn.Module):
    def __init__(self, 
                 object_to_pixel_attention: nn.Module, # object 作为 q
                 self_attention: nn.Module,
                 pixel_to_object_attention: nn.Module,
                 d_model: int=256, # image encoder 出来特征图的C
                 dim_feedforward: int=2048,
                 dropout: float=0.1,
                 ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.object_to_pixel_attention = object_to_pixel_attention
        self.self_attention = self_attention
        self.objectFFN = FFN(dim_in=d_model, dim_ff=dim_feedforward)
        self.pixel_to_object_attention = pixel_to_object_attention
        self.pixelFFN = PixelFFN(dim=d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        
    def forward(self, 
                pixel_input, # BxhxwxC
                object_input, # BxNxC
                pixel_pe,
                object_pe,
                ):
        B, h, w, C = pixel_input.shape
        _, N, _ = object_input.shape
        pixel_input = pixel_input.flatten(1,2).contiguous()
        pixel_pe = pixel_pe.flatten(1, 2).contiguous()
        
        # 1. cross attention: q=object, k=v=pixel
        object_features = self.norm1(object_input)
        object_features_tmp = self.object_to_pixel_attention(
            q=object_features+object_pe,
            k=pixel_input+pixel_pe,
            v=pixel_input
        )
        object_features = object_features + self.dropout1(object_features_tmp)
    
        # 2. self attention: q=k=v=object
        object_features = self.norm2(object_features)
        object_features_tmp = self.self_attention(
            q=object_features+object_pe,
            k=object_features+object_pe,
            v=object_features
        )
        object_features = object_features + self.dropout2(object_features_tmp)
        
        # 3. object FFN
        object_features = self.norm3(object_features)
        object_features_tmp = self.objectFFN(object_features)
        object_features = object_features + self.dropout3(object_features_tmp)
        
        # 4. cross attention: q=pixel, k=v=object
        pixel_features = self.norm4(pixel_input)
        pixel_features_tmp = self.pixel_to_object_attention(
            q=pixel_features+pixel_pe,
            k=object_features+object_pe,
            v=object_features
        )
        pixel_features = pixel_features + self.dropout4(pixel_features_tmp)
        
        # 5. pixel FFN
        pixel_features = self.norm5(pixel_features) # Bx(hw)xC
        pixel_features_tmp = self.pixelFFN(
            pixel=torch.zeros(B, 1, C, h, w),
            pixel_flat=pixel_features
        )
        pixel_features_tmp = pixel_features_tmp.view(B, C, (h*w)).permute(0,2,1)
        pixel_features = pixel_features + self.dropout5(pixel_features_tmp)
        pixel_features = pixel_features.view(B, h, w, C)
        
        return pixel_features, object_features

if __name__ == '__main__':
    B = 4
    h, w = 64, 64
    C = 128
    N = 32 
    layer = ObjectAttentionLayer(d_model=C, 
                                 self_attention=Attention(embedding_dim=C,num_heads=1),
                                 object_to_pixel_attention=Attention(embedding_dim=C,num_heads=1),
                                 pixel_to_object_attention=Attention(embedding_dim=C,num_heads=1)
    )
    pixel_input = torch.rand(B, h, w, C)
    object_input = torch.rand(B, N, C)
    pixel_pe = torch.rand(B, h, w, C)
    object_pe = torch.rand(B, N, C)
    pixel_feature, obejct_feature = layer(pixel_input, object_input, pixel_pe, object_pe)
    print(pixel_feature.shape)
    print(obejct_feature.shape)
    
    
    
        
        
        
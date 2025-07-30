import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from typing import Optional

from .utils.memory_encoder import *
from .utils.utils import *
from .utils.transformer_modules import *
from .utils.position_encoding import *


def SinPositionEncoding(max_sequence_length, d_model, base=10000):
    pe = torch.zeros(max_sequence_length, d_model, dtype=torch.float) 
    exp_1 = torch.arange(d_model // 2, dtype=torch.float)  
    exp_value = exp_1 / (d_model / 2)

    alpha = 1 / (base ** exp_value)  
    out = torch.arange(max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]  
    embedding_sin = torch.sin(out)
    embedding_cos = torch.cos(out)

    pe[:, 0::2] = embedding_sin  
    pe[:, 1::2] = embedding_cos  
    return pe.cuda()


class PixelAttentionLayer(nn.Module):
    def __init__(
        self,
        activation: str = 'relu',
        cross_attention: nn.Module = RoPEAttention(
            rope_theta=10000.0,
            feat_sizes=[32,32],
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1,
            kv_in_dim=64
            ),
        d_model: int=256,
        dim_feedforward: int=2048,
        dropout: float=0.1, # 0.1
        pos_enc_at_attn: bool=False, # F
        pos_enc_at_cross_attn_keys: bool=True, # T
        pos_enc_at_cross_attn_queries: bool=False, # F
        self_attention: nn.Module=RoPEAttention(
            rope_theta=10000.0,
            feat_sizes=[32, 32],
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            dropout=0.1
            ),
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if self.self_attn is not None:
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        
        if self.cross_attn_image is not None: 
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
            
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2, _ = self.self_attn(q, k, v=tgt2, topk_num = None)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0, topk_num=None):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention)
        tgt2 = self.norm2(tgt)
        tgt2, topk_percent = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            topk_num = topk_num,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt, topk_percent
    
    def forward(
        self,
        tgt, # Bx(hw)xC
        memory, # 4096+4, Bx(Thw+To)xc
        pos: Optional[Tensor] = None, 
        query_pos: Optional[Tensor] = None, # 1, 4096, 256
        num_k_exclude_rope: int = 0, # 4
        ca_topk_num = None, 
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        if self.self_attn is not None:
            tgt = self._forward_sa(tgt, query_pos)
    
        if self.cross_attn_image is not None:
            tgt, topk_percent = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope, topk_num=ca_topk_num)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, topk_percent
    
class PixelTransformer(nn.Module):
    def __init__(
        self,
        d_model: int=256,
        pos_enc_at_input: bool=True,
        layer: nn.Module=PixelAttentionLayer(
            activation = 'relu',
            cross_attention = RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[32,32],
                rope_k_repeat=True,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64
                ),
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1, # 0.1
            pos_enc_at_attn=False, # F
            pos_enc_at_cross_attn_keys=True, # T
            pos_enc_at_cross_attn_queries=False, # F
            self_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=[32, 32],
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1
                )
            ),
        num_layers: int = 4,
        batch_first: bool = True,  
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # (hw)xBx256 self-attention inputs
        memory: torch.Tensor,  # (Thw)xBx64 cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # T*4 number of object pointer *tokens*
        ca_topk_num = None,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        if self.batch_first:
            assert (
                curr.shape[1] == memory.shape[1]
            ), "Batch size must be the same for curr and memory"
        else:
            assert (
                curr.shape[0] == memory.shape[0]
            ), "Batch size must be the same for curr and memory"
            
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers: # 4
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output, topk_percent = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                ca_topk_num = ca_topk_num,
                **kwds,
            )
        normed_output = self.norm(output) # 1, 4096, 256 = B, (hw), 256

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output, topk_percent

class PixelMemoryManager(nn.Module):
    def __init__(self, 
                 d_model = 256, 
                 out_dim = 64, 
                 image_h = 64,
                 raw_mask_H = 512,
                 memory_limit = 4,
                 num_layers = 2,
                 num_heads = 1,
                 ca_topk_num = None,
                 pos_enc_at_attn=False, 
                 pos_enc_at_cross_attn_keys=True, 
                 pos_enc_at_cross_attn_queries=False, 
                 cross_attention = "RoPEAttention",
                 ):
        super().__init__()
        if image_h > raw_mask_H:
            raw_mask_H = image_h
        elif raw_mask_H % image_h !=0:
            mul = raw_mask_H//image_h +1
            raw_mask_H = image_h * mul
        assert raw_mask_H % image_h ==0
        total_stride = int(raw_mask_H/image_h)
        
        down_kernel_size=15
        down_stride = 8
        down_padding=7
        if down_stride > total_stride:
            down_stride = total_stride
            down_kernel_size = down_stride*2-1
            down_padding = down_kernel_size - down_stride
        
        if cross_attention == "RoPEAttention":
            self.cross_attention = RoPEAttention(
                    rope_theta=10000.0,
                    feat_sizes=[image_h, image_h],
                    rope_k_repeat=True,
                    embedding_dim=d_model,
                    num_heads=num_heads,
                    downsample_rate=1,
                    dropout=0.1,
                    kv_in_dim=out_dim
                    )
        else:
            self.cross_attention = Attention(
                    embedding_dim=d_model,
                    num_heads=num_heads,
                    downsample_rate=1,
                    dropout=0.1,
                    kv_in_dim=out_dim
                    )
        
        self.memory_encoder = MemoryEncoder(
                     out_dim=out_dim,
                     mask_downsampler = MaskDownSampler(
                         embed_dim = d_model,
                         kernel_size=down_kernel_size, 
                         stride=down_stride, 
                         padding=down_padding, 
                         total_stride=int(raw_mask_H/image_h)), # mask_H / total_stride = pix_feat_H
                     fuser = Fuser(
                                 CXBlock(dim=d_model, kernel_size=7, padding=3, layer_scale_init_value=1e-6,use_dwconv=True),
                                 num_layers=2
                                 ),
                     position_encoding = PositionEmbeddingSine(num_pos_feats=out_dim, normalize=True, scale=None, temperature=10000),
                     in_dim=d_model,  # in_dim of pix_feats
        )
        self.pixel_transformer = PixelTransformer(
            d_model=d_model,
            pos_enc_at_input=True,
            layer=PixelAttentionLayer(
                activation = 'relu',
                cross_attention = self.cross_attention,
                d_model=d_model,
                dim_feedforward=2048,
                dropout=0.1, # 0.1
                pos_enc_at_attn=pos_enc_at_attn, # F
                pos_enc_at_cross_attn_keys=pos_enc_at_attn, # T
                pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries, # F
                self_attention=RoPEAttention(
                    rope_theta=10000.0,
                    feat_sizes=[image_h,image_h],
                    embedding_dim=d_model,
                    num_heads=num_heads,
                    downsample_rate=1,
                    dropout=0.1
                    )
                ),
            num_layers=num_layers,
            batch_first=False
        )
        self.image_position_encoder = PositionEmbeddingSine(
                     num_pos_feats=d_model,
                     normalize=True,
                     scale=None,
                     temperature=10000
        )
        
        self.mask_pixel_feature_bank = None # Bx(Thw)xC and T <= self.memory_limit
        self.mask_pixel_pos_bank = None # Bx(Thw)xC and T <= self.memory_limit
    
        self.memory_limit = memory_limit
        self.ca_topk_num = ca_topk_num
        self.ca_topk_percent_bank = []
    
    def init_loss_func():
        BCE = torch.nn.BCEWithLogitsLoss()
        return BCE
        
    def add_mask_to_memory(self, 
                           image_feature: torch.Tensor, # (B, C, h, w)=(B, 256, 64, 64)
                           mask: torch.Tensor, # (B, 1, H, W)=(B, 1, 512, 512)
                           ):

        # assert mask.shape[-1] / self.memory_encoder.mask_downsampler.total_stride == image_feature.shape[-1]
        
        mask_pixel_feature, mask_pixel_pos = self.memory_encoder(pix_feat=image_feature, masks=mask) # B, out_dim, h, w
        
        B, c, h, w  = mask_pixel_feature.shape
        mask_pixel_feature = mask_pixel_feature.view(B, c, -1) # B, out_dim, (hw)
        mask_pixel_feature = mask_pixel_feature.transpose(1, 2).contiguous() # B, (hw), out_dim
        
        mask_pixel_pos = mask_pixel_pos.view(B, c, -1) # B, out_dim, (hw)
        mask_pixel_pos = mask_pixel_pos.transpose(1, 2).contiguous() # B, (hw), out_dim
        
        if self.mask_pixel_feature_bank is None:
            self.mask_pixel_feature_bank = mask_pixel_feature
            self.mask_pixel_pos_bank = mask_pixel_pos
        else:
            current_num = int(self.mask_pixel_feature_bank.shape[1] / (h*w))

            if current_num == self.memory_limit:
                self.mask_pixel_feature_bank = self.mask_pixel_feature_bank[:, h*w:, :]
                self.mask_pixel_pos_bank = self.mask_pixel_pos_bank[:, h*w:, :]
            self.mask_pixel_feature_bank = torch.cat((self.mask_pixel_feature_bank, mask_pixel_feature), dim=1)
            self.mask_pixel_pos_bank = torch.cat((self.mask_pixel_pos_bank, mask_pixel_pos), dim=1)
    
    def reset(self):
        self.mask_pixel_feature_bank = None
        self.mask_pixel_pos_bank = None
        
    def read_out(self, 
                image_features: torch.Tensor, # BxCxhxw
                ):
        B, C, h, w = image_features.shape
        image_pos = self.image_position_encoder(image_features).to(image_features.dtype) # BxCxhxw
        
        image_features = image_features.view(B, C, h*w).permute(0, 2, 1).contiguous()
        image_pos = image_pos.view(B, C, h*w).permute(0, 2, 1).contiguous()

        output, topk_percent = self.pixel_transformer(
            curr=image_features, # Bx(hw)xC
            curr_pos=image_pos,
            memory=self.mask_pixel_feature_bank, # Bx(Thw)xc
            memory_pos=self.mask_pixel_pos_bank,
            ca_topk_num = self.ca_topk_num
        ) 
        
        output = output.permute(0, 2, 1).view(B, C, h, w) # BxCxhxw
        
        return output
    
    def cal_topk_percent(self):
        if self.ca_topk_num is not None:
            return sum(self.ca_topk_percent_bank)/len(self.ca_topk_percent_bank)
        else:
            return 0.0
    

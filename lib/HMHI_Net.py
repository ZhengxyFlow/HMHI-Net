# This file is modified based on https://github.com/DLUT-yyc/Isomer (OriginalProject)
# Original license: MIT
# Modified 2025

import _init_paths

import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import random
import math


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
                    
class ConvRelu1(nn.Sequential):
    def __init__(self, in_channels: int,\
                out_channels: int,\
                kernel_size: int,\
                stride: int,\
                padding: int,\
                dilation: int,\
                groups: int):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,\
                out_channels=out_channels,\
                kernel_size=kernel_size,\
                stride=stride,\
                padding=padding,\
                dilation=dilation,\
                groups=groups))
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

    
class HMHI_Net(nn.Module):
    def __init__(self, opt):
        super(HMHI_Net, self).__init__()
        self.opt = opt
        self.bn = nn.BatchNorm2d
        self.num_points = opt.num_points

        if opt.encoder == 'swin_tiny':
            self.feature_channels = [96, 192, 384, 768]
            self.img_size = [128, 64, 32, 16]
            embedding_dim = 192
            self.backbone = lib.swin_tiny()
        elif opt.encoder == 'mit_b1':
            self.feature_channels = [64, 128, 320, 512]
            self.img_size = [128, 64, 32, 16]
            embedding_dim = 256
            self.backbone = lib.SMTC_b1()

        self.decode_head = lib.CBAM_Decoder(self.feature_channels, embedding_dim)
        
        self.init_module()
        self.init_backbone()
        
        self.add_mem0 = opt.add_mem0
        self.add_mem1 = opt.add_mem1
        self.add_obj_to_pixel = opt.add_obj_to_pixel
        self.add_pixel_to_obj = opt.add_pixel_to_obj
        
        self.obj_downsacle_num = 1      #Compress semantic memory into object tokens
        self.obj_down_stride = 4
        self.obj_encode = nn.Sequential()
        
        self.image_feature_id = [1,3]   #Serial number of the layers chosen for memory
        assert self.img_size[self.image_feature_id[1]] % self.obj_downsacle_num==0
        obj_size = self.img_size[self.image_feature_id[1]]// self.obj_downsacle_num
        
        if self.add_mem0:
            self.Mem0=lib.PixelMemoryManager(
                d_model=self.feature_channels[self.image_feature_id[0]], 
                out_dim = self.feature_channels[self.image_feature_id[0]], 
                image_h = self.img_size[self.image_feature_id[0]], raw_mask_H = self.img_size[0], \
                memory_limit=opt.train_num_frame,  num_layers=1, num_heads=1, ca_topk_num=50)
            self.mem0_interval = 1
            self.mem0_interval_count = 0
        
        if self.add_mem1:
            self.Mem1=lib.PixelMemoryManager(
                d_model=self.feature_channels[self.image_feature_id[1]], 
                out_dim = self.feature_channels[self.image_feature_id[1]], 
                image_h = obj_size, raw_mask_H = self.img_size[0], memory_limit=opt.train_num_frame, \
                num_layers=1, num_heads=1, pos_enc_at_attn=True, 
                pos_enc_at_cross_attn_keys=False, pos_enc_at_cross_attn_queries=False,\
                cross_attention = "Attention")
            self.mem1_interval = 1
            self.mem1_interval_count = 0 
        
            if self.obj_downsacle_num > 1:
                self.obj_scale_init()
            else:
                print("Semantic memory feature maitain original size")
        
        if self.add_pixel_to_obj:
            assert self.add_mem0 and self.add_mem1, \
                "Pixel_to_obj requires adding two memory banks: mem0 & mem1"
            self.pixel_fea_downsample_init()
        if self.add_obj_to_pixel:
            assert self.add_mem0 and self.add_mem1, \
                "Obj_to_pixel requires adding two memory banks mem0 & mem1"
            self.obj_to_pixel_init()
    
    def init_backbone(self):
        
        if self.opt.encoder == 'swin_tiny':
            saved_state_dict = torch.load('./checkpoint/pretrained/swing_trans/swin_tiny_patch4_window7_224.pth', map_location='cpu')
            saved_state_dict = saved_state_dict['model']
        if self.opt.encoder == 'mit_b1':
            saved_state_dict = torch.load('./checkpoint/pretrained/mit/mit_b1.pth', map_location='cpu')
    
        if 'swin' in self.opt.encoder:
            z1=self.backbone.load_state_dict(saved_state_dict, strict=False) #'model'
        elif 'mit' in self.opt.encoder:
            z1=self.backbone.load_state_dict(saved_state_dict, strict=False)
            # print(f"Successfully Load From Segformer, with keys loaded: {z1}")
    
    def obj_scale_init(self):
        stride = self.obj_down_stride
        
        self.obj_scale_times = int(math.log(self.obj_downsacle_num,stride))
        res_mul = self.obj_downsacle_num // (stride ** self.obj_scale_times)
        self.scale_list =[]
        
        assert self.obj_downsacle_num == math.pow(stride,self.obj_scale_times) * res_mul,\
            f"high level semantic features scale times not appropriate :{self.obj_downsacle_num}"
        
        for i in range(self.obj_scale_times):
            kernel_size = (stride//2)*3
            kernel_size = (kernel_size//2)*2 + 1
            pad_size = int((kernel_size-1)//2)
            print(f"kernel:{kernel_size}, pad:{pad_size}, stride:{stride}")
            assert pad_size*2+1 == kernel_size
            self.obj_encode.append(ConvRelu1(in_channels=self.feature_channels[self.image_feature_id[1]],
                                            out_channels=self.feature_channels[self.image_feature_id[1]],
                                            kernel_size=kernel_size,
                                            stride= stride,
                                            padding=pad_size,
                                            dilation=1,
                                            groups=self.feature_channels[self.image_feature_id[1]]))
            self.scale_list.append(stride)
        if res_mul > 1:
            res_stride = res_mul
            res_kernel = (res_stride//2)*3
            res_kernel = (res_kernel//2)*2 + 1
            res_pad = int((res_kernel-1)//2)
            print(f"Residual kernel:{res_kernel}, pad:{res_pad}, stride:{res_stride}")
            self.obj_encode.append(ConvRelu1(in_channels=self.feature_channels[self.image_feature_id[1]],
                                                out_channels=self.feature_channels[self.image_feature_id[1]],
                                                kernel_size=res_kernel,
                                                stride= res_stride,
                                                padding=res_pad,
                                                dilation=1,
                                                groups=self.feature_channels[self.image_feature_id[1]]))
            self.scale_list.append(res_mul)
        
        self.obj_encode.append(ConvRelu(self.feature_channels[self.image_feature_id[1]], self.feature_channels[self.image_feature_id[1]], 1, 1,0))
    
    def obj_to_pixel_init(self):
        self.obj_reduce  = nn.Sequential()
        self.obj_reduce.append(nn.Linear(in_features=self.feature_channels[self.image_feature_id[1]], out_features=self.feature_channels[self.image_feature_id[0]]))
        self.obj_reduce.append(nn.ReLU())
        
        d_model = self.feature_channels[self.image_feature_id[0]]
        self.obj_to_pixel = lib.PixelTransformer(
            d_model=d_model,
            pos_enc_at_input=True,
            layer=lib.PixelAttentionLayer(
                activation = 'relu',
                cross_attention = lib.Attention(
                    embedding_dim=d_model,
                    num_heads=1,
                    downsample_rate=1,
                    dropout=0.1,
                    kv_in_dim=d_model
                    ),
                d_model=d_model,
                dim_feedforward=2048,
                dropout=0.1, # 0.1
                pos_enc_at_attn=True, # F
                pos_enc_at_cross_attn_keys=False, # T
                pos_enc_at_cross_attn_queries=False, # F
                self_attention=lib.RoPEAttention(
                    rope_theta=10000.0,
                    feat_sizes=[self.img_size[0],\
                                self.img_size[0]],
                    embedding_dim=d_model,
                    num_heads=1,
                    downsample_rate=1,
                    dropout=0.1
                    )
                ),
            num_layers=1,
            batch_first=False
        )
    
    def pixel_fea_downsample_init(self):
        #for f64 downsacle
        self.pixel_down_sample = nn.Sequential(ConvRelu1(in_channels=self.feature_channels[self.image_feature_id[0]],
                                            out_channels=self.feature_channels[self.image_feature_id[1]],
                                            kernel_size=7,
                                            stride= 4,
                                            padding= 3,
                                            dilation=1,
                                            groups=self.feature_channels[self.image_feature_id[0]]),
                                      ConvRelu(self.feature_channels[-1], self.feature_channels[-1], 1, 1, 0))
        self.pixel_to_obj_cbam = CBAM(self.feature_channels[-1]*2)
        self.p2o_down_linear= ConvRelu(self.feature_channels[-1]*2, self.feature_channels[-1], 1, 1, 0)
        
    def init_module(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return 
    
    def obj_down_sample(self, x):
        x = self.obj_encode(x)
        return x
    
    def SGIM_obj_refine_pixel(self, f_p_, f_o):
        f_p = f_p_
        B1, C1, h1, w1 = f_p.shape
        B2, C2, h2, w2 = f_o.shape
        
        pos_p = self.Mem0.image_position_encoder(f_p).to(f_p.dtype) # BxCxhxw
        pos_o = self.Mem1.image_position_encoder(f_o).to(f_o.dtype) # BxCxhxw
        
        f_p = f_p.view(B1, C1, h1*w1).permute(0, 2, 1).contiguous()
        pos_p = pos_p.view(B1, C1, h1*w1).permute(0, 2, 1).contiguous()
        f_o = f_o.view(B2, C2, h2*w2).permute(0, 2, 1).contiguous()
        pos_o = pos_o.view(B2, C2, h2*w2).permute(0, 2, 1).contiguous()
        
        f_o = self.obj_reduce(f_o)
        pos_o = self.obj_reduce(pos_o)
        
        output, _ = self.obj_to_pixel(
            curr=f_p, # Bx(hw)xC
            curr_pos=pos_p,
            memory=f_o, # Bx(Thw)xc
            memory_pos=pos_o,
        ) # Bx(hw)xC
        
        output = output.permute(0, 2, 1).view(B1, C1, h1, w1) # BxCxhxw
        return output
    
    def PLAM_pixel_refine_obj(self, f_p, f_o):
        f_p = self.pixel_down_sample(f_p)
        p_o_f = self.pixel_to_obj_cbam(torch.cat([f_p,f_o],dim=1))
        f_o_ = self.p2o_down_linear(p_o_f)
        return f_o_+f_o
    
    def memorize_or_not0(self):
        if self.mem0_interval_count % self.mem0_interval ==0:
            self.mem0_interval_count = 0
            return True
        else:
            return False
    
    def memorize_or_not1(self):
        if self.mem1_interval_count % self.mem1_interval ==0:
            self.mem1_interval_count = 0
            return True
        else:
            return False
    
    def reset_mem(self):
        if self.add_mem0:
            self.Mem0.reset()
            self.mem0_interval_count = 0
        
        if self.add_mem1:
            self.Mem1.reset()
            self.mem1_interval_count = 0
        return 
        
    def forward(self, imgs, flows, mode):
        
        assert imgs.shape == flows.shape, \
            f"train encoder imgs.shape {imgs.shape} != flows.shape {flows.shape}"
            
        if mode=='train':
            assert len(imgs.shape) == 5,f"imgs.shape {imgs.shape}"  #B,L,C,H,W,

            imgs, flows, img_first, flow_first, first_idx=self.select_first(imgs, flows)
            B, L_s, C,H,W = imgs.shape
            mask_list=[]
            
            #constructing ref
            F_first = self.encode_forward(img_first.squeeze(1), flow_first.squeeze(1))
            M_f, M_raw = self.decode_forward(F_first, return_raw=True)
            
            if self.add_mem0:
                self.Mem0.add_mask_to_memory(F_first[self.image_feature_id[0]], M_raw) 
                self.mem0_interval_count += 1
            
            if self.add_mem1:
                F_first_obj =  self.obj_down_sample(F_first[self.image_feature_id[1]])  
                self.Mem1.add_mask_to_memory(F_first_obj, M_raw)   
                self.mem1_interval_count += 1
            
            mask_list.append(M_f[0])

            for i in range(L_s):
                img = imgs[:,i].squeeze(1)
                flow = flows[:,i].squeeze(1)
                F_i = self.encode_forward(img, flow)
                
                #Memory readout
                if self.add_mem0:
                    ori_Fi_0 = F_i[self.image_feature_id[0]]
                    Fi_0_ = self.Mem0.read_out(ori_Fi_0)
                if self.add_mem1:
                    ori_Fi_1 = F_i[self.image_feature_id[1]]
                    Fi_1_ = self.Mem1.read_out(ori_Fi_1)
                
                if self.add_obj_to_pixel:
                    Fi_0_ref = self.SGIM_obj_refine_pixel(Fi_0_, Fi_1_)
                else:
                    Fi_0_ref = Fi_0_
                    
                if self.add_pixel_to_obj:
                    Fi_1_ref = self.PLAM_pixel_refine_obj(Fi_0_, Fi_1_)
                else:
                    Fi_1_ref = Fi_1_
                    
                if self.add_mem0:
                    F_i[self.image_feature_id[0]] = Fi_0_ref
                if self.add_mem1:
                    F_i[self.image_feature_id[1]] = Fi_1_ref
                    Fi_1_obj =self.obj_down_sample(Fi_1_ref)
                
                M_i, M_raw = self.decode_forward(F_i, return_raw=True)
                mask_list.append(M_i[0])
                
                if self.add_mem0:
                    self.Mem0.add_mask_to_memory(Fi_0_ref, M_raw)   
                    self.mem0_interval_count += 1
                if self.add_mem1:
                    self.Mem1.add_mask_to_memory(Fi_1_obj, M_raw)   
                    self.mem1_interval_count += 1
                
            self.reset_mem()
            return mask_list
        
        else: # eval
            self.reset_mem()
            B, L, C, H, W = imgs.shape
            mask_list=[]
            
            img_first = imgs[:,0]
            flow_first = flows[:,0]
            
            #constructing ref
            F_first = self.encode_forward(img_first.squeeze(1), flow_first.squeeze(1))
            M_f, M_raw = self.decode_forward(F_first, return_raw=True)
            
            if self.add_mem0:
                self.Mem0.add_mask_to_memory(F_first[self.image_feature_id[0]], M_raw) 
                self.mem0_interval_count += 1
            
            if self.add_mem1:
                F_first_obj =  self.obj_down_sample(F_first[self.image_feature_id[1]])  
                self.Mem1.add_mask_to_memory(F_first_obj, M_raw)   
                self.mem1_interval_count += 1
            mask_list.append(M_f[0])
            
            for i in range(L)[1:]:
                img = imgs[:,i]
                flow = flows[:,i]
                F_i = self.encode_forward(img, flow)
                
                #Memory readout
                if self.add_mem0:
                    ori_Fi_0 = F_i[self.image_feature_id[0]]
                    Fi_0_ = self.Mem0.read_out(ori_Fi_0)
                if self.add_mem1:
                    ori_Fi_1 = F_i[self.image_feature_id[1]]
                    Fi_1_ = self.Mem1.read_out(ori_Fi_1)
                
                if self.add_obj_to_pixel:
                    Fi_0_ref = self.SGIM_obj_refine_pixel(Fi_0_, Fi_1_)
                else:
                    Fi_0_ref = Fi_0_
                    
                if self.add_pixel_to_obj:
                    Fi_1_ref = self.PLAM_pixel_refine_obj(Fi_0_, Fi_1_)
                else:
                    Fi_1_ref = Fi_1_
                    
                if self.add_mem0:
                    F_i[self.image_feature_id[0]] = Fi_0_ref
                if self.add_mem1:
                    F_i[self.image_feature_id[1]] = Fi_1_ref
                    Fi_1_obj =self.obj_down_sample(Fi_1_ref)
                
                M_i, M_raw = self.decode_forward(F_i, return_raw=True)
                mask_list.append(M_i[0])
                
                if self.add_mem0 and self.memorize_or_not0():
                    self.Mem0.add_mask_to_memory(Fi_0_ref, M_raw)   
                    self.mem0_interval_count += 1
                if self.add_mem1 and self.memorize_or_not1():
                    self.Mem1.add_mask_to_memory(Fi_1_obj, M_raw)   
                    self.mem1_interval_count += 1
                
            self.reset_mem()
        return mask_list
            
    def encode_forward(self, x, y):
        image_flow=torch.cat([x,y],dim=0)
        
        if_0, if_1, if_2, if_3 = self.backbone(image_flow)
        
        B,_,_,_ = if_0.shape
        x_0 , y_0 = if_0[:B//2], if_0[B//2:]
        x_1 , y_1 = if_1[:B//2], if_1[B//2:]
        x_2 , y_2 = if_2[:B//2], if_2[B//2:]
        x_3 , y_3 = if_3[:B//2], if_3[B//2:]
        
        # BxCxhxw
        z_0 = x_0 + y_0 
        z_1 = x_1 + y_1 
        z_2 = x_2 + y_2 
        z_3 = x_3 + y_3

        return [z_0, z_1, z_2, z_3]
    
    def decode_forward(self, image_flow, return_raw=False):
        if return_raw:
            z, raw_z = self.decode_head(image_flow, return_raw)
            return z, raw_z
        z = self.decode_head(image_flow)
        return z
    
    def select_first(self,imgs,flows):
        #split ref and mem during training stage
        B1,M_n1,C1,H1,W1=imgs.size()              
        B2,M_n2,C2,H2,W2=flows.size()  
        img_list=list(range(M_n1))
        
        # first_idx=random.sample(img_list, 1)
        first_idx=[0]
        
        img_list.pop(first_idx[0])
        self.img_num=len(img_list)
        
        img_ref=imgs[:,first_idx]               
        flow_ref=flows[:,first_idx]               
        imgs=imgs[:,img_list]
        flows=flows[:,img_list]

        return imgs, flows, img_ref, flow_ref, first_idx


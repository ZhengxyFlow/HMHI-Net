# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 d_model=768,
                 d_out=None,
                 bottleneck=64,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck
        self.adapter_type = init_option
        if d_out is None:
            self.d_out = self.n_embd
        else:
            self.d_out = d_out

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.dropout = dropout

        if init_option == "lora":
            self.down_proj = nn.Linear(self.n_embd, self.down_size)
            self.non_linear_func = nn.ReLU()
            self.up_proj = nn.Linear(self.down_size, self.d_out)         
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        
        elif init_option == 'mora':
            self.r = 256
            self.matrix = nn.Linear(self.r, self.r, bias=False)
            with torch.no_grad():
                nn.init.zeros_(self.matrix.weight)

    def forward(self, x, add_residual=False, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        if self.adapter_type == 'lora':
            down = self.down_proj(x)
            down = self.non_linear_func(down)
            down = nn.functional.dropout(down, p=self.dropout, training=self.training)
            up = self.up_proj(down)

            up = up * self.scale

        elif self.adapter_type == 'mora':
            sum_inter = self.n_embd // self.r
            if self.n_embd % self.r != 0:
                pad_size = self.r - self.n_embd % self.r
                x = torch.cat([x, x[..., :pad_size]], dim=-1)
                sum_inter += 1
            in_x = x.view(*x.shape[:-1], sum_inter, self.r).sum(dim=-2) # compress:[B, r]
            out_x = self.matrix(in_x) # matrix: r -> r
            # decompress: because ouput_dim equals input_dim, repeat_time equals sum_inter
            # repeat_time = self.output_dim % self.r
            # if self.output_dim % self.r != 0:
            #     repeat_time += 1
            up = torch.cat([out_x] * sum_inter, dim=-1)[..., :self.n_embd]


        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:  # False
            output = up + residual
        else:
            output = up
        return output
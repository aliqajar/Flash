


import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):

    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # combine Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # gives the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    
    def forward(self, x, causal_mask=False):

        # x: (batch_size, seq_len, dim)

        # (batch_size, seq_len, dim)
        input_shape = x.shape

        # (batch_size, seq_len, dim)    
        batch_size, sequence_length, d_embed = input_shape

        # (batch_size, seq_len, dim/h)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) -> # tensor of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, h, dim/h) -> (batch_size, h, seq_len, dim/h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, h, dim/h) -> (batch_size, h, seq_len, dim/h)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            #mask where the upper triangle is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            #fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)

        
        # divide by d_k (dim/h)
        # (batch_size, h, seq_len, seq_len) -> (batch_size, h, seq_len, seq_len)
        weight /= math.sqrt(self.d_head)

        # (batch_size, h, seq_len, seq_len) -> (batch_size, h, seq_len, seq_len)
        weight = F.softmax(weight, dim =-1)

        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, dim/h) -> (batch_size, h, seq_len, dim/h)
        output = weight @ v

        # (batch_size, h, seq_len, dim/h) -> (batch_size, seq_len, dim)
        output = output.reshape(input_shape)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        output = self.out_proj(output)

        # (batch_size, seq_len, dim)
        return output
    


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):

        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed//n_heads


    def forward(self, x, y):
        # x (latent): (batch_size, seq_len_q, dim_q)
        # y (context): (batch_size, seq_len_kv, dim_kv) = (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # divide each embedding of q into multiple heads such that d_head*n_heads = dim_q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (batch_size, seq_len_q, dim_q) -> (batch_size, seq_len_q, dim_q)
        q = self.q_proj(x)

        # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)
        k = self.k_proj(y)

        # (batch_size, seq_len_kv, dim_kv) -> (batch_size, seq_len_kv, dim_q)
        v = self.v_proj(y)

        # (batch_size, seq_len_q, dim_q) -> (batch_size, seq_len_q, h, dim_q/h) -> (batch_size, h, seq_len_q, dim_q/h)
        q = q.view(interim_shape).transpose(1, 2)
        
        # (batch_size, seq_len_kv, dim_q) -> (batch_size, seq_len_kv, h, dim_q/h) -> (batch_size, h, seq_len_kv, dim_q/h)
        k = k.view(interim_shape).transpose(1, 2)

        # (batch_size, seq_len_kv, dim_q) -> (batch_size, seq_len_kv, h, dim_q/h) -> (batch_size, h, seq_len_kv, dim_q/h)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch_size, h, seq_len_q, dim_q/h) @ (batch_size, h, dim_q/h, seq_len_kv) -> (batch_size, h, seq_len_q, seq_len_kv)
        weight = q @ k.transponse(-1, -2)

        # (batch_size, h, seq_len_q, seq_len_kv)
        weight /= math.sqrt(self.d_head)

        # (batch_size, h, seq_len_q, seq_len_kv)
        weight = F.softmax(weight, dim=-1)
        
        # (batch_size, h, seq_len_q, seq_len_kv) @ (batch_size, h, seq_len_kv, dim_q/h) -> (batch_size, h, seq_len_q, dim_q/h)
        output = weight @ v

        # (batch_size, h, seq_len_q, dim_q/h) -> (batch_size, seq_len_q, h, dim_q/h)
        output = output.transpose(1, 2).contiguous()

        # (batch_size, seq_len_q, h, dim_q/h) -> (batch_size, seq_len_q, dim_q)
        output = output.view(input_shape)

        # (batch_size, seq_len_q, dim_q) -> (batch_size, seq_len_q, dim_q)
        output = self.out_proj(output)

        # (batch_size, seq_len_q, dim_q)
        return output
    

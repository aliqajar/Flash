import os
import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
from transformers import BertTokenizer
from datasets import load_dataset




vocab_size = 30522
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
max_position_embeddings = 512
max_length = 128
batch_size = 32
learning_rate = 5e-5
num_epochs = 1000

class FlashModel(nn.Module):

    def __init__(
            self, 
            vocab_size,
            hidden_size, 
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddigns):
        

        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([
            FlashLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, input_ids, attention_mask):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)
        for layer in self.layers:
            embeddings = layer(embeddings, attention_mask)
        
        return embeddings
    

class FlashLayer(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, intermediate_size):

        super().__init__()

        self.attention = FlashAttention(hidden_size, num_attention_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attenttion(hidden_states, attention_mask)
        residual = hidden_states + attention_output
        layernorm_output = self.layernorm1(residual)
        feed_forward_output = self.feed_forward(layernorm_output)
        residual = layernorm_output + feed_forward_output
        layernorm_output = self.layernorm2(residual)
        return layernorm_output
    


class FlashAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        








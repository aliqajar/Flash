import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import math


class BertModel(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings):

        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])

    
    def forward(self, input_ids, attention_mask):
        seq_length 

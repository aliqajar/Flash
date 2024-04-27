

import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class ClipEmbedding(nn.Module):

    def __init__(self, n_vocab, n_embd, n_token):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)

        # a learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embd))


    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x
    

class ClipLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)

        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x 

        # self attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        # feed-forward layer
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)

        x += residue

        return x
    


class Clip(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = ClipEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([ClipLayer(12, 768) for i in range(12)])
        self.layernorm = nn.LayerNorm(768)


    def forward(self, tokens:torch.LongTensor):
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output
    
    


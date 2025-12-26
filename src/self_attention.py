import torch
import torch.nn as nn
import time

class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

        context_vec = attn_weights @ values
        return context_vec

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.context_length = context_length
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
        print("Attention weights:", attn_weights)
        context_vec = attn_weights @ values
        return context_vec

def __main__():
    d_in = 8
    d_out = 16
    current_time_seconds = time.time()
    torch.manual_seed(current_time_seconds)
    sa_v1 = SelfAttention(d_in, d_out)

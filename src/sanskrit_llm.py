from importlib.metadata import version
import torch
import torch.nn as nn
import tiktoken as tk
import time

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class SanskritLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = LayerNorm(emb_dim=cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


def load_sanskrit_llm_model():
    model = SanskritLLM(GPT_CONFIG_124M)
    return model

if __name__ == "__main__":
    model = load_sanskrit_llm_model()
    tokenizer = tk.get_encoding("gpt2")
    batch = []
    txt1 = "स ते वीर्यं बलं दर्पमुत्सेकं च तथाविधम्। व्यपनेष्यति गात्रेभ्यः शरवर्षेण संयुगे॥"
    txt2 = "स हि देवरसंयुक्तो मम भर्ता महाद्युतिः। निर्भयो वीर्यमाश्रित्य शून्ये वसति दण्डके॥"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch = torch.stack(batch, dim=0)
    print(batch.shape)
    print(batch)
#    print("Batch before padding 1:", batch)
#    print(tokenizer.decode(tokenizer.encode(txt1)))
#    batch.append(torch.tensor(tokenizer.encode(txt2)))
#    print("Batch before padding 2:", batch)
#    batch = torch.stack(batch, dim=0)
#    print(batch)

    # sample_input = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 16))
    torch.manual_seed(time.time())
    output = model(batch)
    print("Output shape:", output.shape)
    print("Output:", output)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_sinusoidal_pe(context_window, emb_dim):
    pos = torch.arange(context_window).unsqueeze(1).float()
    dim = torch.arange(0, emb_dim, 2).float()
    angles = pos / (10000.0 ** (dim / emb_dim))
    pe = torch.zeros(context_window, emb_dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe.unsqueeze(0)  # (1, T, emb_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        return self.resid_drop(self.out_proj(attn_out))


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(emb_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, emb_dim=768, num_heads=12, depth=12,
                 context_window=1024, dropout=0.1):
        super().__init__()
        self.context_window = context_window

        self.backbone = nn.ModuleDict({
            "token_emb": nn.Embedding(vocab_size, emb_dim),
            "drop": nn.Dropout(dropout),
            "blocks": nn.ModuleList(
                [TransformerBlock(emb_dim, num_heads, dropout) for _ in range(depth)]
            ),
            "ln_f": nn.LayerNorm(emb_dim),
        })
        self.backbone.register_buffer("pos_emb", _make_sinusoidal_pe(context_window, emb_dim))
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)

        self.apply(self._init_weights)
        # Residual projection scaling (GPT-2 convention)
        for block in self.backbone.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * depth))
            nn.init.normal_(block.ffn[2].weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * depth))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context_window, \
            f"Sequence length {T} exceeds context window {self.context_window}"

        tok_emb = self.backbone.token_emb(idx)      # (B, T, D)
        pos_emb = self.backbone.pos_emb[:, :T, :]   # (1, T, D)
        x = self.backbone.drop(tok_emb + pos_emb)

        for block in self.backbone.blocks:
            x = block(x)

        x = self.backbone.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

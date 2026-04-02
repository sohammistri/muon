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


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, q_input, kv_input, attn_mask=None, is_causal=False):
        B, T_q, D = q_input.shape
        T_kv = kv_input.shape[1]

        q = self.q_proj(q_input).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T_q, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.resid_drop(self.out_proj(attn_out))


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.self_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_mask=None):
        normed = self.ln1(x)
        x = x + self.self_attn(normed, normed, attn_mask=src_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.self_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.cross_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.ln3 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, enc_out, causal_pad_mask=None, cross_mask=None):
        normed = self.ln1(x)
        x = x + self.self_attn(normed, normed, attn_mask=causal_pad_mask)
        normed = self.ln2(x)
        x = x + self.cross_attn(normed, enc_out, attn_mask=cross_mask)
        x = x + self.ffn(self.ln3(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, num_heads=8, depth=6,
                 context_window=256, dropout=0.1, pad_id=None):
        super().__init__()
        self.context_window = context_window
        self.pad_id = pad_id
        num_embeddings = vocab_size + 1 if pad_id is not None else vocab_size
        out_features = num_embeddings

        self.backbone = nn.ModuleDict({
            "token_emb": nn.Embedding(num_embeddings, emb_dim, padding_idx=pad_id),
            "drop": nn.Dropout(dropout),
            "encoder_blocks": nn.ModuleList(
                [EncoderBlock(emb_dim, num_heads, dropout) for _ in range(depth)]
            ),
            "decoder_blocks": nn.ModuleList(
                [DecoderBlock(emb_dim, num_heads, dropout) for _ in range(depth)]
            ),
            "enc_ln_f": nn.LayerNorm(emb_dim),
            "dec_ln_f": nn.LayerNorm(emb_dim),
        })
        self.backbone.register_buffer("pos_emb", _make_sinusoidal_pe(context_window, emb_dim))
        self.head = nn.Linear(emb_dim, out_features, bias=False)

        self.apply(self._init_weights)
        # Residual projection scaling (GPT-2 convention)
        for blocks in [self.backbone.encoder_blocks, self.backbone.decoder_blocks]:
            for block in blocks:
                nn.init.normal_(block.self_attn.out_proj.weight, mean=0.0,
                                std=0.02 / math.sqrt(2 * depth))
                nn.init.normal_(block.ffn[3].weight, mean=0.0,
                                std=0.02 / math.sqrt(2 * depth))
        for block in self.backbone.decoder_blocks:
            nn.init.normal_(block.cross_attn.out_proj.weight, mean=0.0,
                            std=0.02 / math.sqrt(2 * depth))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src_ids, tgt_input, src_pad_mask=None, tgt_pad_mask=None):
        """
        Args:
            src_ids:      (B, T_src) encoder input token IDs
            tgt_input:    (B, T_tgt) decoder input token IDs
            src_pad_mask: (B, T_src) bool, True = real token
            tgt_pad_mask: (B, T_tgt) bool, True = real token
        Returns:
            logits: (B, T_tgt, vocab_size+1)
        """
        B, T_src = src_ids.shape
        _, T_tgt = tgt_input.shape
        assert T_src <= self.context_window, \
            f"Source length {T_src} exceeds context window {self.context_window}"
        assert T_tgt <= self.context_window, \
            f"Target length {T_tgt} exceeds context window {self.context_window}"

        # Shared embeddings + sinusoidal PE
        src_emb = self.backbone.drop(
            self.backbone.token_emb(src_ids) + self.backbone.pos_emb[:, :T_src, :]
        )
        tgt_emb = self.backbone.drop(
            self.backbone.token_emb(tgt_input) + self.backbone.pos_emb[:, :T_tgt, :]
        )

        # Build masks
        # Encoder self-attention & cross-attention: mask out PAD keys
        src_mask = None
        if src_pad_mask is not None:
            src_mask = torch.where(
                src_pad_mask[:, None, None, :], 0.0, float('-inf')
            )  # (B, 1, 1, T_src)

        # Decoder self-attention: combined causal + PAD mask
        causal_mask = torch.triu(
            torch.full((T_tgt, T_tgt), float('-inf'), device=src_ids.device),
            diagonal=1,
        )  # (T_tgt, T_tgt)
        if tgt_pad_mask is not None:
            tgt_key_mask = torch.where(
                tgt_pad_mask[:, None, None, :], 0.0, float('-inf')
            )  # (B, 1, 1, T_tgt)
            causal_pad_mask = causal_mask[None, None, :, :] + tgt_key_mask  # (B, 1, T_tgt, T_tgt)
        else:
            causal_pad_mask = causal_mask[None, None, :, :]  # (1, 1, T_tgt, T_tgt)

        # Encoder
        enc_out = src_emb
        for block in self.backbone.encoder_blocks:
            enc_out = block(enc_out, src_mask=src_mask)
        enc_out = self.backbone.enc_ln_f(enc_out)

        # Decoder
        dec_out = tgt_emb
        for block in self.backbone.decoder_blocks:
            dec_out = block(dec_out, enc_out, causal_pad_mask=causal_pad_mask, cross_mask=src_mask)
        dec_out = self.backbone.dec_ln_f(dec_out)

        return self.head(dec_out)  # (B, T_tgt, vocab_size+1)

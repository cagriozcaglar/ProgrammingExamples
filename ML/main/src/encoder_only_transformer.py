import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# -------------------------------
# Positional Encoding
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -------------------------------
# Transformer Encoder Block
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# -------------------------------
# Transformer Encoder Model
# -------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, heads=4, dim_ff=512, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, heads, dim_ff) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)
        return self.output(x)


# -------------------------------
# Dummy Masked Language Modeling Training
# -------------------------------
def generate_fake_batch(batch_size, seq_len, vocab_size, mask_token_id):
    input_ids = torch.randint(5, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    mask = torch.rand(input_ids.shape) < 0.15  # 15% chance to mask
    input_ids[mask] = mask_token_id
    labels[~mask] = -100  # Ignore loss on non-masked positions

    return input_ids, labels


# -------------------------------
# Training Loop
# -------------------------------
def train():
    vocab_size = 1000
    mask_token_id = 3
    model = TransformerEncoder(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for step in range(100):
        x, labels = generate_fake_batch(batch_size=32, seq_len=64, vocab_size=vocab_size, mask_token_id=mask_token_id)
        logits = model(x)  # [B, T, V]
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

if __name__ == "__main__":
    train()
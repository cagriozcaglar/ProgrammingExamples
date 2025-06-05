import torch
import torch.nn as nn
import torch.nn.functional as F
import math

### 1. Transformer Model for Machine Translation

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, emb_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, emb_size)
        return x + self.pe[:, :x.size(1)]

# Token + Positional Embeddings
class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size)

    def forward(self, x):
        return self.pos_encoder(self.token_embed(x))

# Transformer-based Seq2Seq
class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=512, num_heads=8, num_layers=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.src_embedding = Embeddings(src_vocab_size, emb_size)
        self.tgt_embedding = Embeddings(tgt_vocab_size, emb_size)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # src, tgt: (batch, seq_len)
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(output)

# Helper to create causal mask for decoder
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


### 3. Dummy Training Loop

# Dummy dataset
src_vocab_size = tgt_vocab_size = 1000
pad_idx = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerMT(src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_one_batch(model, optimizer, criterion):
    model.train()
    src = torch.randint(2, src_vocab_size, (32, 20)).to(device)
    tgt = torch.randint(2, tgt_vocab_size, (32, 20)).to(device)

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

    logits = model(src, tgt_input, tgt_mask=tgt_mask)

    loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Training loop (simplified)
for epoch in range(10):
    loss = train_one_batch(model, optimizer, criterion)
    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
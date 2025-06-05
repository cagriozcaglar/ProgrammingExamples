import torch
import torch.nn as nn
import torch.nn.functional as F

### 1. Define the Decoder-Only Transformer
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, d_ff=1024, n_layers=4, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))  # Learnable positional embeddings
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).expand(B, -1, -1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


### 2. Toy Dataset – Character-Level Language Modeling
import string
import random

# Example: tiny dataset of sequences using lowercase letters
chars = string.ascii_lowercase + " "
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

def encode(s): return [stoi[c] for c in s]
def decode(t): return ''.join([itos[i] for i in t])

# Generate synthetic data
def generate_data(num_samples=1000, seq_len=32):
    data = []
    for _ in range(num_samples):
        s = ''.join(random.choices(chars, k=seq_len + 1))
        data.append((torch.tensor(encode(s[:-1])), torch.tensor(encode(s[1:]))))
    return data

dataset = generate_data()


### 3. Training the Model
model = DecoderOnlyTransformer(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

EPOCHS = 10
BATCH_SIZE = 16

for epoch in range(EPOCHS):
    random.shuffle(dataset)
    total_loss = 0
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i+BATCH_SIZE]
        x_batch = torch.stack([x for x, y in batch]).to(device)
        y_batch = torch.stack([y for x, y in batch]).to(device)

        logits = model(x_batch)
        loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataset):.4f}")


### 4. Sampling from the Model
def sample(model, start_text="a", length=50):
    model.eval()
    idx = torch.tensor([encode(start_text)], dtype=torch.long).to(device)
    for _ in range(length):
        logits = model(idx)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return decode(idx[0].tolist())

print(sample(model, start_text="the "))
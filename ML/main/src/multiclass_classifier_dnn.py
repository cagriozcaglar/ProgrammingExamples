import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Dummy dataset loading (replace this with real dataset)
df = pd.read_csv('your_dataset.csv')  # should contain 'product_name', 'product_description', 'product_category'

# Preprocessing
df['text'] = df['product_name'].fillna('') + ' ' + df['product_description'].fillna('')
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['product_category'])  # 0 to 199

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ProductDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Dataset and DataLoader
dataset = ProductDataset(df['text'].tolist(), df['label'].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
class ProductClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ProductClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProductClassifier(num_classes=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
def train(model, dataloader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

train(model, dataloader, optimizer, criterion, device)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Load and preprocess data
# df = pd.read_csv('../data/notifications_log.csv')
import os
current_directory = os.getcwd()
ml_main_root = current_directory + "/ML/main/"
data_path = ml_main_root + "data/"
model_path = ml_main_root + "model/"
df_raw = pd.read_csv(data_path + 'notifications_log.csv')
df_raw
'''
>>> df_raw
   user_id  notification_type  hour_of_day  device_type location   engaged
0        1              promo           10      android       NY         1
1        2           reminder           15          ios       CA         0
2        3  birthday_reminder           18          ios       IL         1
3        4              promo           10      android       MA         0
'''

# One-hot encode categorical features
pd.set_option('display.max_columns', None)
df = pd.get_dummies(df_raw, columns=['notification_type', 'device_type', 'location'])
df
'''
>>> df
   user_id  hour_of_day  engaged  notification_type_birthday_reminder  \
0        1           10        1                                False   
1        2           15        0                                False   
2        3           18        1                                 True   
3        4           10        0                                False   

   notification_type_promo  notification_type_reminder  device_type_android  \
0                     True                       False                 True   
1                    False                        True                False   
2                    False                       False                False   
3                     True                       False                 True   

   device_type_ios  location_CA  location_IL  location_MA  location_NY  
0            False        False        False        False         True  
1             True         True        False        False        False  
2             True        False         True        False        False  
3            False        False        False         True        False  
'''

# Split features and labels
X = df.drop(columns=['user_id', 'engaged']).values
y = df['engaged'].values
'''
>>> X
array([[10, False, True, False, True, False, False, False, False, True],
       [15, False, False, True, False, True, True, False, False, False],
       [18, True, False, False, False, True, False, True, False, False],
       [10, False, True, False, True, False, False, False, True, False]],
      dtype=object)
>>> y
array([1, 0, 1, 0])
'''

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)
'''
>>> X
array([[-0.95065415, -0.57735027,  1.        , -0.57735027,  1.        ,
        -1.        , -0.57735027, -0.57735027, -0.57735027,  1.73205081],
       [ 0.5118907 , -0.57735027, -1.        ,  1.73205081, -1.        ,
         1.        ,  1.73205081, -0.57735027, -0.57735027, -0.57735027],
       [ 1.38941761,  1.73205081, -1.        , -0.57735027, -1.        ,
         1.        , -0.57735027,  1.73205081, -0.57735027, -0.57735027],
       [-0.95065415, -0.57735027,  1.        , -0.57735027,  1.        ,
        -1.        , -0.57735027, -0.57735027,  1.73205081, -0.57735027]])
'''

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)
'''
>>> len(X_train)
2
>>> len(X_test)
2
>>> len(y_train)
2
>>> len(y_test)
2
'''

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
'''
>>> X_train_tensor
tensor([[-0.9507, -0.5774,  1.0000, -0.5774,  1.0000, -1.0000, -0.5774, -0.5774,
         -0.5774,  1.7321],
        [-0.9507, -0.5774,  1.0000, -0.5774,  1.0000, -1.0000, -0.5774, -0.5774,
          1.7321, -0.5774]])
>>> y_train_tensor
tensor([[1.],
        [0.]])
>>> X_test_tensor
tensor([[ 1.3894,  1.7321, -1.0000, -0.5774, -1.0000,  1.0000, -0.5774,  1.7321,
         -0.5774, -0.5774],
        [ 0.5119, -0.5774, -1.0000,  1.7321, -1.0000,  1.0000,  1.7321, -0.5774,
         -0.5774, -0.5774]])
>>> y_test_tensor
tensor([[1.],
        [0.]])
'''

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
'''
>>> train_dataset
<torch.utils.data.dataset.TensorDataset object at 0x146dac190>
>>> train_loader
<torch.utils.data.dataloader.DataLoader object at 0x146db9520>
'''

# 2. Define the model
class NotificationModel(nn.Module):
    def __init__(self, input_dim):
        super(NotificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = NotificationModel(input_dim=X_train.shape[1])
'''
>>> model
NotificationModel(
  (model): Sequential(
    (0): Linear(in_features=10, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=64, out_features=32, bias=True)
    (4): ReLU()
    (5): Linear(in_features=32, out_features=1, bias=True)
    (6): Sigmoid()
  )
)
'''

# 3. Set up training
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
'''
>>> criterion
BCELoss()
>>> optimizer
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
'''

# 4. Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
'''
....
NotificationModel(
  (model): Sequential(
    (0): Linear(in_features=10, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=64, out_features=32, bias=True)
    (4): ReLU()
    (5): Linear(in_features=32, out_features=1, bias=True)
    (6): Sigmoid()
  )
)
Epoch 20/20 - Loss: 0.5620
'''

# 5. Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).numpy()
    preds_label = (preds > 0.5).astype(int)

    acc = accuracy_score(y_test, preds_label)
    auc = roc_auc_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC-ROC:  {auc:.4f}")
'''
Test Accuracy: 1.0000
Test AUC-ROC:  1.0000
'''

# 6. Save model
torch.save(model.state_dict(), model_path + 'notification_nn_model.pt')
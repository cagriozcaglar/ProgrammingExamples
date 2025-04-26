import torch
from torch import nn
from d2l import torch as d2l


### Implementation from scratch
class LSTMScratch(d2l.Module):
    def __init__(self, num_inputs: int, num_hiddens: int, sigma: int = 0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (
            init_weight(num_inputs, num_hiddens),
            init_weight(num_inputs, num_hiddens),
            nn.Parameter(torch.zeros(num_hiddens))
        )

        # Input gate
        self.W_xi, self.W_hi, self.b_i = triple()
        # Forget gate
        self.W_xf, self.W_hf, self.b_f = triple()
        # Output gate
        self.W_xf, self.W_hf, self.b_f = triple()
        # Input node
        self.W_xc, self.W_hc, self.b_c = triple()

@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Init state with shape (batch_size, num_hiddens)
        H = torch.zeros(inputs.shape[1], self.num_hiddens, device = inputs.device)
        C = torch.zeros(inputs.shape[1], self.num_hiddens, device = inputs.device)
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        # Input gate
        I = torch.sigmoid(
            torch.matmul(X, self.W_xi) + \
            torch.matmul(H, self.W_hi) + \
            self.b_i
        )
        # Forward gate
        I = torch.sigmoid(
            torch.matmul(X, self.W_xf) + \
            torch.matmul(H, self.W_hf) + \
            self.b_f
        )
        # Output gate
        I = torch.sigmoid(
            torch.matmul(X, self.W_xo) + \
            torch.matmul(H, self.W_ho) + \
            self.b_o
        )
        # C_tilde
        C_tilde = torch.tanh(
            torch.matmul(X, self.W_xc) + \
            torch.matmul(H, self.W_hc) + \
            self.b_c            
        )

        # Calculations
        C = F * C + I * C_tilde 
        H = O * torch.tanh(C)
        outputs.append(H)

        return outputs, (H, C)

# Training and Prediction
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)

### Concise Implementation
class LSTM(d2l.RNN):
    def __init__(self, num_inputs: int, num_hiddens: int):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = LSTM(num_inputs, num_hiddens)
    
    def forward(self, inputs, H_C = None):
        return self.rnn(inputs, H_C)

lstm = LSTM(num_inputs=len(data.vocab), lr=4)
model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
model.predict('it has', 20, data.vocab, d2l.try_gpu())
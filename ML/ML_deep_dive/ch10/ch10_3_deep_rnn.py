import torch
from torch import nn
from d2l import torch as d2l

# Implementation from scratch
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs: int, num_hiddens: int, num_layers: int, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # Deep RNN: num_inputs -> num_hiddens -> num_hiddens -> ... -> num_hiddens
        self.rnn = nn.Sequential(
            *[
                d2l.RNNScratch(
                    num_inputs=num_inputs if i==0 else num_hiddens,
                    num_hiddens=num_hiddens,
                    sigma=sigma,
                )
                for i in range(num_layers)
            ]
        )

@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs: int, Hs=None):
    outputs = inputs
    if Hs is None:
        Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = torch.stack(outputs, 0)
    return outputs, Hs

# Train + Predict
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn_block = StackedRNNScratch(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)

# Concise implementation
class GRU(d2l.RNN):
    # Multi-layer GRU model
    def __init__(self, num_inputs: int, num_hiddens: int, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers, dropout=dropout)

# Train
gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)

# Predict
model.predict('it has', 20, data.vocab, d2l.try_gpu())
import torch
from torch import nn
from d2l import torch as d2l

# Implementation from scratch
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs: int, num_hiddens: int, sigma: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        # Output dimension is doubled
        self.num_hiddens *= 2


@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs else (None, None)
    # Forward RNN: takes inputs in original order, takes f_H
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    # IMPORTANT: Backward RNN: takes inputs in **reversed** order, takes f_H
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    # Concatenate forward and backward RNN outputs
    outputs = [
        torch.cat((f, b), -1)
        # Do not forget to reverse outputs of backward RNN
        for f, b in zip(f_outputs, reversed(b_outputs))
    ]
    return outputs, (f_H, b_H)


# Concise Implementation
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs: int, num_hiddens: int):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        # Torch has nn.GRU with optional bidirectional=True parameter
        # to help with bidirectional RNN implementation
        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
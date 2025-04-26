from torch import nn
from d2l import torch as d2l

# Encoder-Decoder Architecture: Only the interfaces

# Encoder (interface)
class Encoder(nn.Module):
    # Base encoder interface for Encoder-Decoder architecture
    def __init__(self):
        super().__init__()

    # Forward method, to be implemented by subclasses inheriting Encoder interface
    # *args, because there can be additional arguments (e.g. length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
    

# Decoder (interface)
class Decoder(nn.Module):
    # Base decoder interface for Encoder-Decoder architecture
    def __init__(self):
        super().__init__()

    # Later, there can be additional arguments (e.g. length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError


# Encoder-Decoder
class EncoderDecoder(d2l.Classifier):
    # Base class for Encoder-Decoder architecture
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only. This actually calls Decoder.forward method
        return self.decoder(dec_X, dec_state)[0]
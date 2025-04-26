import math
import torch
from torch import nn
from d2l import torch as d2l

# 11.3.1. Dot Product Attention => Masked Softmax Attention
def masked_softmax(X, valid_lens):  #@save
    '''
    Perform softmax operation by masking elements on the last axis.
    '''
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

X = torch.rand(2, 2, 4)
X
# tensor([[[0.2999, 0.0582, 0.3601, 0.2332],
#          [0.9609, 0.9930, 0.4192, 0.8438]],

#         [[0.0080, 0.2383, 0.4317, 0.7191],
#          [0.7590, 0.1379, 0.2606, 0.4831]]])
valid_lens = torch.tensor([2, 1])
valid_lens
# tensor([2, 3])
Y = masked_softmax(X, valid_lens)
Y
# tensor([[[0.5601, 0.4399, 0.0000, 0.0000],
#          [0.4920, 0.5080, 0.0000, 0.0000]],

#         [[0.2641, 0.3325, 0.4034, 0.0000],
#          [0.4662, 0.2505, 0.2832, 0.0000]]])
Y.size()
# torch.Size([2, 2, 4])


X = torch.rand([2,3,4])
X
maxlen=X.size(1)
maxlen
# 3
X = torch.rand((maxlen))
X
# tensor([0.0275, 0.0265, 0.9362])
valid_len=torch.randint(0,2,[2,3,4])
valid_len
# tensor([[[0, 1, 1, 1],
#          [0, 0, 1, 0],
#          [0, 0, 0, 1]],

#         [[0, 0, 0, 1],
#          [1, 1, 0, 0],
#          [1, 0, 0, 0]]])

mask = X[None, :] < valid_len[:, None]
mask
# tensor([[[[False,  True,  True,  True],
#           [False, False,  True, False],
#           [False, False, False,  True]],

#          [[False,  True,  True,  True],
#           [False, False,  True, False],
#           [False, False, False,  True]]],


#         [[[False, False, False,  True],
#           [ True,  True, False, False],
#           [ True, False, False, False]],

#          [[False, False, False,  True],
#           [ True,  True, False, False],
#           [ True, False, False, False]]]])

mask_value = 0
X[~mask] = mask_value
X
# tensor([[[0.1822, 0.7448, 0.3477, 0.1067],
#          [0.8638, 0.5935, 0.1770, 0.3635],
#          [0.5756, 0.4744, 0.9071, 0.6046]],

#         [[0.9262, 0.6955, 0.9379, 0.3220],
#          [0.5681, 0.3244, 0.3474, 0.4994],
#          [0.5123, 0.1891, 0.0827, 0.9345]]])

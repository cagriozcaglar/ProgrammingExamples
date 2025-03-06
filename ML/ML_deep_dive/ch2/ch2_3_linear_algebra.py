import torch

# Scalars
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
# (tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))

# Vectors
x = torch.arange(3)
x
# tensor([0, 1, 2])

x[2]    # tensor(2)
len(x)  # 3
x.shape # torch.Size([3])

# Matrices
A = torch.arange(6).reshape(3, 2)
A
'''
tensor([[0, 1],
        [2, 3],
        [4, 5]])
'''

A.T   # Transpose of matrix A
'''
tensor([[0, 2, 4],
        [1, 3, 5]])
'''

# Symmetric matrices, A == A.T
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
'''
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
'''

# Tensors
torch.arange(24).reshape(2, 3, 4)
'''
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
'''

# Tensor Arithmetic
# Element-wise Ops
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
'''
(
 tensor([[0., 1., 2.],
        [3., 4., 5.]]),
 tensor([[ 0.,  2.,  4.],
        [ 6.,  8., 10.]])
)
'''

# Element-wise multiplication: Hadamard product
A*B
'''
tensor([[ 0.,  1.,  4.],
        [ 9., 16., 25.]])
'''

# Add / Multiply scalar with tensor
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
'''
(
 tensor([[[ 2,  3,  4,  5],
         [ 6,  7,  8,  9],
         [10, 11, 12, 13]],

        [[14, 15, 16, 17],
         [18, 19, 20, 21],
         [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4])
 )
'''

# Reduction
# Sum of a tensor's elements
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
'''
(
 tensor([0., 1., 2.]),
 tensor(3.)
)
'''

A
'''
tensor([[0., 1., 2.],
        [3., 4., 5.]])
'''

A.shape, A.sum()
# (torch.Size([2, 3]), tensor(15.))

# Sum over all elements along the rows (axis = 0)
A, A.sum(axis=0)
'''
(tensor([[0., 1., 2.],
        [3., 4., 5.]]),
 tensor([3., 5., 7.]))
'''
A.shape, A.sum(axis=0).shape
# (torch.Size([2, 3]), torch.Size([3])) # Reduces row dimension, left with 3 columns

# Sum over all elements along the columns (axis = 1)
A, A.sum(axis=1)
'''
(tensor([[0., 1., 2.],
        [3., 4., 5.]]),
 tensor([ 3., 12.]))
'''
A.shape, A.sum(axis=1).shape
# (torch.Size([2, 3]), torch.Size([2]))

# Reduce a matrix along both dimensions (row, column), returns the sum of all elements.
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
# tensor(True)

# Mean value
A.mean(), A.sum() / A.numel()
# (tensor(2.5000), tensor(2.5000))

# Mean value along axis=0
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
# (tensor([1.5000, 2.5000, 3.5000]), tensor([1.5000, 2.5000, 3.5000]))

# Non-Reduction Sum
# Keep the number of axes unchanged when invoking reduction function, do it with keepdims=True
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
# (tensor([[ 3.],
#         [12.]]), torch.Size([2, 1]))

# Usefule when broadcasting which requires size match
A / sum_A
# tensor([[0.0000, 0.3333, 0.6667],
#         [0.2500, 0.3333, 0.4167]])

# Cumulative sum along an axis
A.cumsum(axis=0)
# tensor([[0., 1., 2.],
#         [3., 5., 7.]])

# Dot product
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
# (tensor([0., 1., 2.]),
#  tensor([1., 1., 1.]),
#  tensor(3.))

# Dot product == performing an elementwise multiplication followed by a sum
torch.sum(x * y)
# tensor(3.)

# *Matrix-Vector product*:
# 1) Use mv() function, or (mv: matrix-vector)
# 2) Use @ operator, which can be used for both matrix-vector and matrix-matrix products
A, x, A.shape, x.shape, torch.mv(A, x), A@x
# (tensor([[0., 1., 2.],
#         [3., 4., 5.]]),
#  tensor([0., 1., 2.]),
#  torch.Size([2, 3]),
#  torch.Size([3]),
#  tensor([ 5., 14.]),
#  tensor([ 5., 14.]))

# Matrix-Matrix products
# 1) Use mm() function, or (mv: matrix-matrix)
# 2) Use @ operator, which can be used for both matrix-vector and matrix-matrix products
B = torch.ones(3, 4)
A, B, torch.mm(A, B), A@B
'''
(tensor([[0., 1., 2.],
        [3., 4., 5.]]),
 tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]),
 tensor([[ 3.,  3.,  3.,  3.],
        [12., 12., 12., 12.]]),
tensor([[ 3.,  3.,  3.,  3.],
        [12., 12., 12., 12.]]))
'''

# Norms
# L2-norm:
u = torch.tensor([3.0, -4.0])
u, torch.norm(u)
# (tensor([ 3., -4.]), tensor(5.))
# L1-norm:
u, torch.norm(u, 1), torch.abs(u).sum()
# (tensor([ 3., -4.]), tensor(7.), tensor(7.))
# Frobenius norm: X_F = sqrt(sum_i=1^m sum_j=1^n x_ij^2)
torch.norm(torch.ones((4, 9))) # Explanation: sqrt(sum_squares of all elements in matrix)
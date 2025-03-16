import torch

# Differentiate function y = 2 * x^T * x, w.r.t x
x = torch.arange(4.0)
x
# tensor([0., 1., 2., 3.])
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
# y = 2 * x^T * x
y = 2 * torch.dot(x, x)
# tensor(28., grad_fn=<MulBackward0>)
# Take gradient of y wrt x , by calling backward method
y.backward()
x.grad
# tensor([ 0.,  4.,  8., 12.])
# We know dy / dx = 4*x, check equality
x.grad == 4 * x
# tensor([True, True, True, True])

# Gradient of another function
x.grad.zero_()  # Reset the gradient (important, otherwise buffer will be non-empty)
# tensor([0., 0., 0., 0.])
y = x.sum()
# tensor(6., grad_fn=<SumBackward0>)
y.backward()
x.grad
# tensor([1., 1., 1., 1.])

# Backward for non-scalar variables
x.grad.zero_()
# tensor([0., 0., 0., 0.])
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
# tensor([0., 2., 4., 6.])

# Detach computation
x.grad.zero_()
# tensor([0., 0., 0., 0.])
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
# tensor([True, True, True, True])

x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
# tensor([True, True, True, True])

# Gradient and Python Control Flow
# Sample function
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
# We call this function, passing in a random value, as input. Since the input is a random variable,
# we do not know what form the computational graph will take. However, whenever we execute f(a) on
# a specific input, we realize a specific computational graph and can subsequently run backward.
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
a, d
# (tensor(0.9534, requires_grad=True), tensor(1952.6190, grad_fn=<MulBackward0>))
# Check equality
a.grad == d / a
# tensor(True)


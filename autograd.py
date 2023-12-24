import torch

# Simple 1-layer NN
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

#print(f"Gradient function for z = {z.grad_fn}")
#print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(f"W Gradient: {w.grad}")
print(f"B Gradient: {b.grad}")

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Jacobian product
inp = torch.eye(4, 5, requires_grad=True)
print(f"Inp: \n{inp}\n")
out = (inp+1).pow(2).t()
print(f"Out: \n{out}\n")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

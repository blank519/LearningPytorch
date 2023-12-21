import torch
import numpy as np

# Tensor from array
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
#print(f"Basic tensor: \n {x_data} \n")

# Tensor from np array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
#print(f"Tensor from NumPy array: \n {x_np} \n")

# Ones tensor from other tensor (retains properties of shape and datatype from parameter tensor)
x_ones = torch.ones_like(x_data)
#print(f"Ones tensor from basic tensor: \n {x_ones} \n")

# Random tensor from other tensor (while overriding datatype)
x_rand = torch.rand_like(x_data, dtype=torch.float)
#print(f"Random tensor from basic tensor: \n {x_rand} \n")

shape = (3, 4,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
#print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Ones Tensor: \n {ones_tensor} \n")
#print(f"Zeros Tensor: \n {zeros_tensor}\n")

# Shape can be tuple or inputs
tensor = torch.rand(2, 3, 4)
#print(f"New Random Tensor: \n {tensor}\n")
#print(f"Shape of tensor: {tensor.shape}")
#print(f"Datatype of tensor: {tensor.dtype}")
#print(f"Device tensor is stored on: {tensor.device}\n")

#print(f'First set: \n{tensor[0]}\n')
#print(f'Second set first column: \n{tensor[1, :, 0]}\n')
#print(f'Last column: \n{tensor[..., -1]}\n')

tensor[:,1] = 0
#print(f"Modified Tensor: \n {tensor}\n")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
#print(t1)

data = [[1, 2, 3], [4, 5, 6]]


tensor = torch.tensor(data) 
# Matrix multiplication (all 3 operations return the same matrix)
y1 = tensor @ tensor.T

y2 = tensor.matmul(tensor.T)

y3 = torch.ones_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

#print(f"Multiplication method 1: \n{y1}")
#print(f"Multiplication method 2: \n{y2}")
#print(f"Multiplication method 3: \n{y3}\n")

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.ones_like(tensor)
torch.mul(tensor, tensor, out=z3)
#print(f"Multiplication method 1: \n{z1}")
#print(f"Multiplication method 2: \n{z2}")
#print(f"Multiplication method 3: \n{z3}\n")

agg = tensor.sum()
#print("Sum: {agg}")

agg_item = agg.item()
#print(agg_item, type(agg_item))
#print()
tensor.add_(4)
#print(tensor)


t1 = torch.ones(5)
print(f"t1: {t1}")
n1 = t1.numpy()
print(f"n1: {n1}")
t1.add_(1)
print(f"t1: {t1}")
print(f"n1: {n1}")


n2 = np.ones(5)
t2 = torch.from_numpy(n2)
np.add(n2, 1, out=n2)
print(f"t2: {t2}")
print(f"n2: {n2}")




import time
import torch
import os


a = torch.tensor([[1,2,3],[2,3,5]])
b = torch.tensor([[1,2,3],[2,3,5]])
c = a/b

d= torch.sum(c)
print(d)
print(type(d))
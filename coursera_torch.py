#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:12:07 2020

@author: labuser
"""
import numpy as np
import torch
import pandas as pd

a = [[11, 12, 13], [21,  22, 23], [31, 32, 33]]
A = torch.tensor(a)
A.ndimension()
A.shape
A.size()
A.numel()
A = torch.tensor([[0,1,1], [1,0,1]])
B = torch.tensor([[1,1], [1,1], [-1,1]])
C = torch.mm(A,B)
df = pd.DataFrame({'A':[11,33,22], 'B':[3,3,2]})

# Derivatives
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
x.grad

x = torch.tensor(2.0, requires_grad=True)
z = x**2 + 2*x + 1
z.backward()
x.grad

# Partial derivatives
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u*v + u**2
f.backward()
u.grad

# Test questions
q = torch.tensor(1.0, requires_grad=True)
fq = 2*q**3 + q
fq.backward()
q.grad



# Simple Data Set
from torch.utils.data import Dataset

class toy_set(Dataset):
    def __init__(self, length=100, transform=None):
        self.x = 2*torch.ones(length,2)
        self.y = torch.ones(length,1)
        self.len = length
        self.transform = transform
        
    # Assuming this is like referencing (dataset[0])
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.len
    
dataset = toy_set()
dataset[0]

for i in range(3):
    x,y = dataset[i]
    print(i, 'x:',x, 'y:',y)

class add_mult(object):
    def __init__(self, addx=1, muly=1):
        self.addx = addx
        self.muly = muly
    
    # Assume this is calling the object instance as a function
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample
    
a_m = add_mult()
dataset_ = toy_set(transform=a_m)

class mult(object):
    def __init__(self, mul=100):
        self.mul = mul
        
    def __call__(self,sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mul
        y = y * self.mul
        sample = x, y
        return sample
    
from torchvision import transforms

# applies add_mult() followed by applies mult()
data_transforms = transforms.Compose([add_mult(), mult()])



# 1.5 Dataset

from PIL import Image
import pandas as pd
import os
from matplotlib.pyplot import imshow
from torch.utils.data import  Dataset, DataLoader

directory = "/home/labuser/Documents/fashion_mnist"
csv_file = 'fashion-mnist_train.csv'
csv_path = os.path.join(directory, csv_file)

data_name = pd.read_csv(csv_path)
data_name.head()



# Week 2

# Linear Regression
import torch
w = torch.tensor(2.0, requires_grad=True) # weights
b = torch.tensor(-1.0, requires_grad=True) # bias
def forward(x):
    y = w*x + b
    return y
x = torch.tensor([1.0]) # feature
yhat = forward(x)
yhat

x = torch.tensor([[1],[2]]) # feature vector
yhat = forward(x)

from torch.nn import Linear
torch.manual_seed(1) # slope & bias can be randomly initialized s.t. we get the same results every time

model = Linear(in_features=1, out_features=1) # create model object
# in_features is size of input vectors; out_features is size of output vectors
print(list(model.parameters())) # view randomly-generated model parameters
# first is slope and second is bias

x = torch.tensor([0.0])
yhat = model(x)
yhat

x = torch.tensor([[1.0], [2.0]])
yhat = model(x)
yhat









































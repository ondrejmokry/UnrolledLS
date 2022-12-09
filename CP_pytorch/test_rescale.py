# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:37:39 2022

@author: Ondrej
"""

import torch
from utils import rescale

b = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=torch.float)
a = 2*b + 1

c1, scale1, shift1 = rescale(b, a)

c2 = torch.zeros(b.shape)
scale2 = torch.zeros(2)
shift2 = torch.zeros(2)
for i in range(b.shape[0]):
    c2[i,:,:], scale2[i], shift2[i] = rescale(b[i,:,:].unsqueeze(0), a[i,:,:].unsqueeze(0))
    
print("Difference of scales: " + str(torch.norm(scale1-scale2).item()))
print("Difference of shifts: " + str(torch.norm(shift1-shift2).item()))

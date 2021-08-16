import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.rand(2,3,4)

# left_pad = nn.ConstantPad1d([2*2,0],0)
# out1 = left_pad(input)

right_pad = nn.ConstantPad1d([-2,2*3],0)
out2 = right_pad(input)

print(input)
# print(out1)
print(out2)

bs,dim,_ = input.size()

# left_pad = torch.zeros(bs,dim,4)
# print(left_pad.shape)
# out1 = torch.cat((left_pad,input),dim=2)
# print(out1)

right_pad = torch.zeros(bs,dim,2*3)
print(input[:,2:,:].shape)
print(right_pad.shape)
out2 = torch.cat((input[:,:,2:],right_pad),dim=2)
print(out2)
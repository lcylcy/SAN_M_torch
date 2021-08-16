import torch
import torch.nn as nn

padded_input = torch.rand(3,5,8)
input_lengths = torch.IntTensor([3,2,4])
N = padded_input.size(0)
non_pad_mask = padded_input.new_ones(padded_input.size()[:-1]) 
print(non_pad_mask)
print(non_pad_mask.shape)
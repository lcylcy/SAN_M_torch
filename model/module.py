import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):         
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)                
        self.w_2 = nn.Linear(d_ff, d_model)                 
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, x):
        residual = x                       
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
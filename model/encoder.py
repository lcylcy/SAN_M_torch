import torch.nn as nn

from model.attention import Sub_layer
from model.module import PositionwiseFeedForward
from model.pad_mask_utils import get_non_pad_mask, get_attn_pad_mask


class Encoder(nn.Module):
    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, pe_maxlen=5000):
        super(Encoder, self).__init__()
        self.d_input = d_input         
        self.n_layers = n_layers       
        self.n_head = n_head          
        self.d_k = d_k                 
        self.d_v = d_v                 
        self.d_model = d_model        
        self.d_inner = d_inner        
        self.dropout_rate = dropout    
        self.pe_maxlen = pe_maxlen     

        self.linear_in = nn.Linear(d_input, d_model)      
        self.layer_norm_in = nn.LayerNorm(d_model)         

        self.dropout = nn.Dropout(dropout)        

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])


    def forward(self, padded_input, input_lengths, return_attns=False):

        enc_slf_attn_list = []
   
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)    #(3,5,40),(3)
  
        length = padded_input.size(1)
        

        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length) #(3,5,40),(3),5
        
        
        enc_output = self.layer_norm_in(self.linear_in(padded_input))


        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,                    
                non_pad_mask=non_pad_mask,      
                slf_attn_mask=slf_attn_mask,  
                input_lengths = input_lengths)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output, 


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attn = Sub_layer(
            n_head, d_model, d_k, d_v, dropout=dropout,d_inner = d_inner)  

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)           


    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None,input_lengths=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask,input_lengths=input_lengths)   

        enc_output *= non_pad_mask

        enc_output_out = self.pos_ffn(enc_output)  

        enc_output_out *= non_pad_mask

        return enc_output_out, enc_slf_attn

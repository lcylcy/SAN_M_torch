#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.attention import MultiHeadAttention
from model.module import  PositionwiseFeedForward
from model.pad_mask_utils import (IGNORE_ID, get_attn_key_pad_mask, get_attn_pad_mask,
                                       get_non_pad_mask, get_subsequent_mask, pad_list)
from model.fsmn import DFSMN_undirection_layer


class Decoder(nn.Module):
    def __init__(
            self, sos_id, eos_id,
            n_tgt_vocab, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            pe_maxlen=5000):
        super(Decoder, self).__init__()

        self.sos_id = sos_id           
        self.eos_id = eos_id             
        self.n_tgt_vocab = n_tgt_vocab   
        self.d_word_vec = d_word_vec    
        self.n_layers = n_layers         
        self.n_head = n_head             
        self.d_k = d_k                  
        self.d_v = d_v                   
        self.d_model = d_model         
        self.d_inner = d_inner          
        self.dropout = dropout          
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing 
        self.pe_maxlen = pe_maxlen     

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)                
        #self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack1 = nn.ModuleList([
            DecoderLayer1(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)    
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)   

        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            self.tgt_word_prj.weight = self.tgt_word_emb.weight  
            self.x_logit_scale = (d_model ** -0.5)             
        else:
            self.x_logit_scale = 1.

    def preprocess(self, padded_input):

        ys = [y[y != IGNORE_ID] for y in padded_input]

        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        max_len = padded_input.shape[1] + 2
        ys_in_pad = pad_list(ys_in, self.eos_id, override_max_len=max_len)
        ys_out_pad = pad_list(ys_out, IGNORE_ID, override_max_len=max_len)
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad


    def forward(self, padded_input, encoder_padded_outputs,encoder_input_lengths, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        ys_in_pad, ys_out_pad = self.preprocess(padded_input)    

        non_pad_mask = get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,    
                                                     seq_q=ys_in_pad,    
                                                     pad_idx=self.eos_id)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
  
        output_length = ys_in_pad.size(1)
  
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs,
                                              encoder_input_lengths,
                                              output_length)
        dec_output = self.tgt_word_emb(ys_in_pad) * self.x_logit_scale 


        for dec_layer in self.layer_stack1:

            dec_output, dec_enc_attn = dec_layer(
                dec_output,                           
                encoder_padded_outputs,               
                non_pad_mask=non_pad_mask,            
                slf_attn_mask=slf_attn_mask,          
                dec_enc_attn_mask=dec_enc_attn_mask) 

        seq_logit = self.tgt_word_prj(dec_output)

        pred, gold = seq_logit, ys_out_pad

        if return_attns:
            return pred, gold, dec_slf_attn_list, dec_enc_attn_list
        return pred, gold

    def recognize_beam(self, encoder_outputs, beam, nbest, max_decode_len, text_tokenizer, verbose=False):
        maxlen = max_decode_len         
        if max_decode_len == 0:
            maxlen = encoder_outputs.size(0)

        encoder_outputs = encoder_outputs.unsqueeze(0)   

        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long() 


        hyp = {'score': 0.0, 'yseq': ys, 'word_confidences': [1.0]}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']                                        

                word_confidences = hyp['word_confidences']
                
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) 
                slf_attn_mask = get_subsequent_mask(ys)
                                
                dec_output = self.tgt_word_emb(ys) * self.x_logit_scale

                for dec_layer in self.layer_stack1:
                    dec_output, _ = dec_layer(
                        dec_output, encoder_outputs,
                        non_pad_mask=non_pad_mask,
                        slf_attn_mask=slf_attn_mask,
                        dec_enc_attn_mask=None)

                seq_logit = self.tgt_word_prj(dec_output[:, -1])

                local_scores = F.log_softmax(seq_logit, dim=1)

                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = dict()
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['word_confidences'] = word_confidences + [np.exp(local_best_scores[0, j].cpu().numpy())]
                    new_hyp['yseq'] = torch.ones(1, (1+ys.size(1))).type_as(encoder_outputs).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])

                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]
            hyps = hyps_best_kept

            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = torch.cat([hyp['yseq'],
                                             torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()], dim=1)
                    hyp['word_confidences'].append(1.0)


            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if verbose:
                if len(hyps) > 0:
                    print('remeined hypothes: ' + str(len(hyps)))
                else:
                    print('no hypothesis. Finish decoding.')
                    break
                for hyp in hyps:
                    tokenids = [int(x) for x in hyp['yseq'][0, 1:]]
                    text = text_tokenizer.tokens_to_text(tokenids)
                    print("hype:{}".format(text))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), nbest)]
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
        return nbest_hyps


class DecoderLayer1(nn.Module):
  
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer1, self).__init__()

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout) 
        self.slf_attn = DFSMN_undirection_layer(d_model, d_model, d_inner,20,1,d_model)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) 


    def forward(self,
                dec_input,               
                enc_output,              
                non_pad_mask=None,      
                slf_attn_mask=None,       
                dec_enc_attn_mask=None  
                ):
        out1  = self.pos_ffn(dec_input)

        out1 *= non_pad_mask
        
        out2 = self.slf_attn(out1)

        out2 *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(out2,                   
                                                 enc_output,             
                                                 enc_output,             
                                                 mask=dec_enc_attn_mask) 

        dec_output *= non_pad_mask

        return dec_output , dec_enc_attn



class DecoderLayer2(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer2, self).__init__()

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.slf_attn = DFSMN_undirection_layer(d_model, d_model, d_inner,20,1)  

        self.layer_norm1 = nn.LayerNorm(d_model)


    def forward(self,
                dec_input,             
                enc_output,               
                non_pad_mask=None,       
                slf_attn_mask=None,     
                dec_enc_attn_mask=None   
                ):


        out1  = self.pos_ffn(dec_input)

        out1 *= non_pad_mask

        out1 = self.slf_attn(out1)

        out1 *= non_pad_mask

        return out1

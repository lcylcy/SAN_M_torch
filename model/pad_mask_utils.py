import torch
IGNORE_ID = -1

def pad_list(xs, pad_value, override_max_len=0):
    n_batch = len(xs)
    max_len = max(max(x.size(0) for x in xs), override_max_len)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    return non_pad_mask.unsqueeze(-1)



def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    return padding_mask


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1]) 
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    return non_pad_mask.unsqueeze(-1)


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) 
    return subsequent_mask
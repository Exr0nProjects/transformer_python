# 21XIPR
# attempt at GPT-2 infrence implementation

import numpy as np
from numpy.random import default_rng
import math

try:
    from torch import nn
    import torch
except ImportError:
    print('you need pytorch! https://pytorch.org/get-started/locally/')
    exit(1)

from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model

config = {
    # 'size': {
    #     'emb_dim': 768,     # (attn_heads = 12) * (attn_dim = 64)
    #     'stack_height': 12,
    #     'attn_heads': 12,
    #     'ctx_size': 1024,
    #     'vocab_size': 50257,
    #     }
    'size': {
        'emb_dim': 4,     # (attn_heads = 12) * (attn_dim = 64)
        'stack_height': 12,
        'attn_heads': 2,
        'ctx_size': 1024,
        'vocab_size': 50257,
        }
    }

def get_shapes():
    sz = config['size']
    return {
            'embedding': {
                'positional': (sz['ctx_size'], sz['emb_dim']),
                'token': (sz['vocab_size'], sz['emb_dim']),
            },
            'decoder': [{
                'attn': {
                    'query_w': (sz['emb_dim'], sz['emb_dim']),
                    'query_b': (sz['emb_dim'], 1),
                    'key_w':   (sz['emb_dim'], sz['emb_dim']),
                    'key_b':   (sz['emb_dim'], 1),
                    'value_w': (sz['emb_dim'], sz['emb_dim']),
                    'value_b': (sz['emb_dim'], 1),
                    'proj_w':  (sz['emb_dim'], sz['emb_dim']),
                    'proj_b':  (sz['emb_dim'], 1),
                },
                'dense': {
                    'fc_w':    (sz['emb_dim']*4, sz['emb_dim']),
                    'fc_b':    (sz['emb_dim']*4, 1),
                    'proj_w':  (sz['emb_dim'], sz['emb_dim']*4),
                    'proj_b':  (sz['emb_dim'], 1),
                }
            }]*sz['stack_height']
        }


def initialize(shapes):
    it = shapes.items() if type(shapes) is dict else enumerate(shapes)
    for k, v in it:
        if type(v) is dict or type(v) is list:
            shapes[k] = initialize(v)
        elif type(v) is tuple:
            shapes[k] = rng.standard_normal(v)  # TODO: weights in each layer are the same
    return shapes


def nonlin(x: np.ndarray):  # NTFS: convert to in place operation
    # return x * (x > 0)    # relu
    from scipy.special import erf
    return x * 0.5 * (1 + erf(x/np.sqrt(2)))   # gelu


def softmax(x: np.ndarray):     # NTFS: convert to in place operation
    ret = np.exp(x)
    sums = np.sum(ret, 1)       # NTFS: new memory, or just use stack mem

    for row, div in zip(ret, sums):
        row /= div

    return ret


def layernorm(x: np.ndarray):
    sums = x.sum(axis=0)
    means = sums/x.shape[0]
    std = x.std(axis=0)
    for i, row in enumerate(x):
        x[i] = (row - means) / std
    return x


def feed_forward(dense, x):
    assert(x.shape == (config['size']['emb_dim'], seq_len))
    norm_x = layernorm(x)   # TODO: should layernorm happen before or after skip input
    return nonlin(dense['proj_b'] + dense['proj_w'].dot(
           nonlin(dense[  'fc_b'] + dense[  'fc_w'].dot(norm_x))     # NTFS: new memory initialized (emb_dim * 4, seq_len)
    )) + x


def self_attention(attn, x):
    # Adds the mask so decoder cannot see values after current word
    def mask(x):
        new_mat = x
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                new_mat[i, j] = -math.inf
        return new_mat

    assert(x.shape == (config['size']['emb_dim'], seq_len))

    adim = config['size']['emb_dim'] // config['size']['attn_heads']

    norm_x = layernorm(x) # TODO: should the norm happen before or after skip connection input? graykode says after, and this implements after

    # NTFS: new memory initialized
    gq = attn['query_w'].dot(norm_x) + attn['query_b']
    gk = attn['key_w'  ].dot(norm_x) + attn['key_b'  ]
    gv = attn['value_w'].dot(norm_x) + attn['value_b']

    for hdi in range(config['size']['attn_heads']):     # NTFS: loop can be parallelized
        bq, bk, bv = gq[hdi*adim:(hdi+1)*adim], gk[hdi*adim:(hdi+1)*adim], gv[hdi*adim:(hdi+1)*adim]
        scalars = bq.T.dot(bk) / np.sqrt(adim)
        scalars = mask(scalars)
        scalars = softmax(scalars)

        for j, s in enumerate(scalars):
            gq[hdi*adim:(hdi+1)*adim, j] = np.sum(s * bv, 1)     # NTFS: writing to gq is a memory saving technique

    return attn['proj_w'].dot(gq) + attn['proj_b'] + x

def infer(inp: np.ndarray):
    for decoder in mats['decoder']:
        attn, dense = decoder.values()
        x = self_attention(attn, inp)
        inp = feed_forward(dense, x)
        print(inp)
    return inp


# BEGIN YOINKS from andrew
#   class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  emb_size: int,
#                  dropout: float,
#                  maxlen: int = 5000):
#         super(PositionalEncoding, self).__init__()
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)
#
#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)
#
#     def forward(self, token_embedding: Tensor):
#         return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
#
# # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
# class TokenEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, emb_size):
#         super(TokenEmbedding, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.emb_size = emb_size
#
#     def forward(self, tokens: Tensor):
#         return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
# END YOINKS

rng = default_rng(1333)
seq_len = 5

if __name__ == '__main__':
    mats = initialize(get_shapes())
    # print(mats)

    sz = config['size']
    gpt2config = GPT2Config(
            vocab_size=sz['vocab_size'],
            n_positions=sz['ctx_size'],
            n_ctx=sz['ctx_size'],
            n_embd=sz['emb_dim'],
            n_layer=sz['stack_height'],
            n_head=sz['attn_heads'],
            n_inner=4)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')   # this should be gpt2 small?
    pretrained = GPT2Model.from_pretrained('gpt2')

    inp = "I like my sandwhiches with peanut butter and"
    toks = tokenizer(inp, return_tensors='pt')
    print(toks)

    got = pretrained(**toks, output_hidden_states=True)
    print(got.hidden_states)

    # pretrained_hidden_states =

    # inp = rng.normal(0, 0.0002, (config['size']['emb_dim'], seq_len))
    # print(inp)



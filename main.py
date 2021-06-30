# 21XIPR
# attempt at GPT-2 infrence implementation

import numpy as np
from numpy.random import default_rng

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
        'stack_height': 2,
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


def feed_forward(dense, x):
    assert(x.shape == (config['size']['emb_dim'], seq_len))
    # TODO: layernorm
    return nonlin(dense['proj_b'] + dense['proj_w'].dot(
           nonlin(dense[  'fc_b'] + dense[  'fc_w'].dot(x))     # NTFS: new memory initialized (emb_dim * 4, seq_len)
    )) + x


def self_attention(attn, x):
    assert(x.shape == (config['size']['emb_dim'], seq_len))

    adim = config['size']['emb_dim'] // config['size']['attn_heads']

    # NTFS: new memory initialized
    gq = attn['query_w'].dot(x) + attn['query_b']
    gk = attn['key_w'  ].dot(x) + attn['key_b'  ]
    gv = attn['value_w'].dot(x) + attn['value_b']

    for hdi in range(config['size']['attn_heads']):     # NTFS: loop can be parallelized
        bq, bk, bv = gq[hdi*adim:(hdi+1)*adim], gk[hdi*adim:(hdi+1)*adim], gv[hdi*adim:(hdi+1)*adim]
        scalars = bq.T.dot(bk) / np.sqrt(adim)
        scalars = softmax(scalars)

        for j, s in enumerate(scalars):
            gq[hdi*adim:(hdi+1)*adim, j] = np.sum(s * bv, 1)     # NTFS: writing to gq is a memory saving technique

    return attn['proj_w'].dot(gq) + attn['proj_b'] + x


rng = default_rng(1336)
seq_len = 5

if __name__ == '__main__':
    mats = initialize(get_shapes())
    # print(mats)

    inp = rng.standard_normal((config['size']['emb_dim'], seq_len))
    # print(inp)

    for decoder in mats['decoder']:
        attn, dense = decoder.values()
        x = self_attention(attn, inp)
        inp = feed_forward(dense, x)
        # print(inp)

# 21XIPR
# attempt at GPT-2 infrence implementation

config = {
    'size': {
        'emb_dim': 768,     # (attn_heads = 12) * (attn_dim = 64)
        'stack_height': 12,
        'attn_heads': 12,
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
                    'query_b': (sz['emb_dim'], sz['emb_dim']),
                    'key_w':   (sz['emb_dim'], sz['emb_dim']),
                    'key_b':   (sz['emb_dim'], sz['emb_dim']),
                    'value_w': (sz['emb_dim'], sz['emb_dim']),
                    'value_b': (sz['emb_dim'], sz['emb_dim']),
                    'proj_w':  (sz['emb_dim'], sz['emb_dim']),
                    'proj_b':  (sz['emb_dim'], sz['emb_dim']),
                },
                'dense': {
                    'fc_w':    (sz['emb_dim']*4, sz['emb_dim']),
                    'fc_b':    (sz['emb_dim']*4, sz['emb_dim']),
                    'proj_w':  (sz['emb_dim'], sz['emb_dim']*4),
                    'proj_b':  (sz['emb_dim'], sz['emb_dim']*4),
                }
            }]*sz['stack_height']
        }

import numpy as np

def initialize(shapes):
    from numpy.random import default_rng
    rng = default_rng(1336)
    it = shapes.items() if type(shapes) is dict else enumerate(shapes)
    for k, v in it:
        if type(v) is dict or type(v) is list:
            shapes[k] = initialize(v)
        elif type(v) is tuple:
            shapes[k] = rng.standard_normal(v)
    return shapes

if __name__ == '__main__':
    mats = initialize(get_shapes())
    print(mats)

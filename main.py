# 21XIPR
# attempt at GPT-2 infrence implementation

config = {
    'size': {
        'embedding_dim': 768,
        'num_blocks': 12,
        'attn_heads': 12,
        'ctx_size': 1024,
        }
    }

shapes = {
        'embedding': {
            'ctx': (config['size']['ctx_size'], config['size']['embedding_dim']),
            }
        }

import numpy as np

def initialize(shapes):
    from numpy.random import default_rng
    rng = default_rng(1336)
    it = shapes.items() if type(shapes) is dict else enumerate(shapes)
    for k, v in it:
        if type(v) is dict or type(v) is list:
            v = initialize(v)
        elif type(v) is tuple:
            shapes[k] = rng.standard_normal(v)
        else:
            raise NotImplementedError('allocation description must be a dict of shape-tuples')
    return shapes

if __name__ == '__main__':
    mats = initialize(shapes)
    print(mats)

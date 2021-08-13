# 21XIPR
# attempt at GPT-2 infrence implementation

import numpy as np
from numpy.random import default_rng
import math
from copy import deepcopy

try:
    from torch import nn
    import torch
except ImportError:
    print('you need pytorch! https://pytorch.org/get-started/locally/')
    exit(1)

!pip install transformers
from transformers import GPT2Config, GPT2TokenizerFast, GPT2Model


config = {
     'size': {
         'emb_dim': 768,     # (attn_heads = 12) * (attn_dim = 64)
         'stack_height': 12,
         'attn_heads': 12,
         'ctx_size': 1024,
         'vocab_size': 50257,
         }
    #'size': {
     #   'emb_dim': 4,     # (attn_heads = 12) * (attn_dim = 64)
      #  'stack_height': 12,
       # 'attn_heads': 2,
        #'ctx_size': 1024,
        #'vocab_size': 50257,
        #}
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
                    'ln1_w': (sz['emb_dim'], 1),
                    'ln1_b': (sz['emb_dim'], 1),
                    'attn_w': (3*sz['emb_dim'], sz['emb_dim']),
                    'attn_b': (sz['emb_dim'], 1),
                    'proj_w':  (sz['emb_dim'], sz['emb_dim']),
                    'proj_b':  (sz['emb_dim'], 1),
                },
                'dense': {
                    'ln2_w': (sz['emb_dim'], 1),
                    'ln2_b': (sz['emb_dim'], 1),
                    'fc_w':    (sz['emb_dim']*4, sz['emb_dim']),
                    'fc_b':    (sz['emb_dim']*4, 1),
                    'proj_w':  (sz['emb_dim'], sz['emb_dim']*4),
                    'proj_b':  (sz['emb_dim'], 1),
                }
            }]*sz['stack_height']
        }

def casually_masked_softmax(x: np.ndarray):     # NTFS: convert to in place operation
    ret = np.exp(x)

    for i in range(seq_len):
        for j in range(i+1, seq_len):
            ret[i][j] = 0
    sums = np.sum(ret, 1)       # NTFS: new memory, or just use stack mem

    for row, div in zip(ret, sums):
        row /= div

    return ret

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


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    x = np.transpose(x)
    assert(x.shape == (seq_len, config['size']['emb_dim']))
    sums = x.sum(axis=1)

    means = sums/x.shape[1]
    stdsq = x.std(axis=1).reshape(11,1)
    stdsq = np.square(stdsq)
    means = means.reshape(11,1)
    x = (x-means)/np.sqrt(stdsq+1e-5)
    x = w*x+b

    return np.transpose(x)


'''def feed_forward(dense, x):
    assert(x.shape == (config['size']['emb_dim'], seq_len))
    norm_x = layernorm(x, dense['ln2_w'], dense['ln2_b'])   # TODO: should layernorm happen before or after skip input
    #print("Normed")
    #print((nonlin(dense[  'fc_b'] + dense[  'fc_w'].dot(norm_x)).shape))
    return nonlin(dense['proj_b'] + dense['proj_w'].dot(
           nonlin(dense[  'fc_b'] + dense[  'fc_w'].dot(norm_x))     # NTFS: new memory initialized (emb_dim * 4, seq_len)
    )) + x'''

def feed_forward(dense, x):
    assert(x.shape == (config['size']['emb_dim'], seq_len))
    norm_x = layernorm(x, dense['ln2_w'], dense['ln2_b'])   # TODO: should layernorm happen before or after skip input
    h = nonlin(dense[  'fc_b'] + dense[  'fc_w'].dot(norm_x))
    h2 = dense['proj_b'] + dense['proj_w'].dot(h)
    return h2 + x

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

    norm_x = layernorm(x, attn['ln1_w'], attn['ln1_b'])

    # NTFS: new memory initialized
    at = attn['attn_w'].dot(norm_x) + attn['attn_b']
    gq, gk, gv = np.vsplit(at, 3)
  
    for hdi in range(12):#range(config['size']['attn_heads']):     # NTFS: loop can be parallelized
        bq, bk, bv = gq[hdi*adim:(hdi+1)*adim], gk[hdi*adim:(hdi+1)*adim], gv[hdi*adim:(hdi+1)*adim]
        scalars = bk.T.dot(bq) / np.sqrt(adim)
        scalars = np.transpose(casually_masked_softmax(np.transpose(scalars)))
        gq[hdi*adim:(hdi+1)*adim] = bv.dot(scalars)

    return attn['proj_w'].dot(gq) + attn['proj_b'] + x

def infer(inp: np.ndarray):
    i = 0
    for decoder in mats['decoder']:
        attn, dense = decoder.values()
        x = self_attention(attn, inp)
        inp = feed_forward(dense, x)
    return inp

rng = default_rng(1333)
seq_len = 11

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

    mats['decoder'] = [deepcopy(a) for a in mats['decoder']]

    for head in range(0, 12):
          #print(pretrained.state_dict()["h."+str(head)+".mlp.c_proj.bias"].numpy()[0])
          mats['decoder'][head]['attn']['ln1_b'] = pretrained.state_dict()["h."+str(head)+".ln_1.bias"].numpy()#.reshape(-1,1)
          mats['decoder'][head]['attn']['ln1_w'] = pretrained.state_dict()["h."+str(head)+".ln_1.weight"].numpy()
          mats['decoder'][head]['attn']['attn_w'] = pretrained.state_dict()["h."+str(head)+".attn.c_attn.weight"].T.numpy()
          mats['decoder'][head]['attn']['attn_b'] = pretrained.state_dict()["h."+str(head)+".attn.c_attn.bias"].numpy().reshape(-1, 1)
          mats['decoder'][head]['attn']['proj_b'] = pretrained.state_dict()["h."+str(head)+".attn.c_proj.bias"].numpy()
          mats['decoder'][head]['attn']['proj_w'] = pretrained.state_dict()["h."+str(head)+".attn.c_proj.weight"].T.numpy()
          mats['decoder'][head]['attn']['proj_b'] = mats['decoder'][head]['attn']['proj_b'].reshape(-1, 1)
          #Dense stuff
          mats['decoder'][head]['dense']['ln2_b'] = pretrained.state_dict()["h."+str(head)+".ln_2.bias"].numpy()#.reshape(-1,1)
          print(head)
          print(mats['decoder'][head]['dense']['ln2_b'][0:5])
          mats['decoder'][head]['dense']['ln2_w'] = pretrained.state_dict()["h."+str(head)+".ln_2.weight"].numpy()
          mats['decoder'][head]['dense']['proj_b'] = pretrained.state_dict()["h."+str(head)+".mlp.c_proj.bias"].numpy()
          mats['decoder'][head]['dense']['proj_w'] = pretrained.state_dict()["h."+str(head)+".mlp.c_proj.weight"].T.numpy()
          mats['decoder'][head]['dense']['proj_b'] = mats['decoder'][head]['dense']['proj_b'].reshape(-1, 1)
          mats['decoder'][head]['dense']['fc_b'] = pretrained.state_dict()["h."+str(head)+".mlp.c_fc.bias"].numpy()
          mats['decoder'][head]['dense']['fc_w'] = pretrained.state_dict()["h."+str(head)+".mlp.c_fc.weight"].T.numpy()
          mats['decoder'][head]['dense']['fc_b'] = mats['decoder'][head]['dense']['fc_b'].reshape(-1,1)

    print(pretrained.state_dict()["h.11.ln_2.bias"].numpy()[0:5])

    for head in range(0, 12):
      mats['decoder'][head]['dense']['ln2_b'] = np.copy(pretrained.state_dict()["h."+str(head)+".ln_2.bias"].numpy())
      print(head)
      print(mats['decoder'][head]['dense']['ln2_b'][0:5])
    for head in range(0, 12):
      print(head)
      print(mats['decoder'][head]['dense']['ln2_b'][0:5])

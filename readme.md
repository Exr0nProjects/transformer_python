# transformers python

A GPT-2 implementation using as vanilla python as possible, for future conversion to C++.

## steps

- [ ] implement the decoder stack (masked self-attention, feed forward) using numpy
- [ ] implement transformer things (positional encoding)
- [ ] standardize weight-loading schema, test to ensure that it's actually working
- [ ] tally used numpy methods and implement a naive matrix class that supports thoes operations

## carried by
- [Jay Alammar: The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/#part-2-illustrated-self-attention)

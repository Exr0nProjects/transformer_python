# transformers python

A GPT-2 implementation using as vanilla python as possible, for future conversion to C++.

## steps

- [ ] implement the decoder stack (masked self-attention, feed forward) using numpy
- [ ] implement transformer things (positional encoding)
- [ ] standardize weight-loading schema, test to ensure that it's actually working
- [ ] tally used numpy methods and implement a naive matrix class that supports thoes operations

## carried by
- [Jay Alammar: The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/#part-2-illustrated-self-attention)
- [GPT-2 paper section 2.3](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT paper section 4.1 Model specifications](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Attention Is All You Need section 3.1](https://arxiv.org/pdf/1706.03762.pdf)

"""The full model (plan Step 10).

tokens -> token embedding + positional encoding -> dropout
       -> N blocks -> final LayerNorm -> linear head -> logits over the vocabulary
"""

from transformer.block import Block
from transformer.layers import Dropout, Embedding, Layer, LayerNorm, Linear, sinusoidal_positions


class Transformer(Layer):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, rng,
                 dropout=0.0, causal=True):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = Embedding(vocab_size, d_model, rng)
        self.positions = sinusoidal_positions(max_len, d_model)  # fixed, not trained
        self.emb_drop = Dropout(dropout, rng)
        self.blocks = [Block(d_model, n_heads, rng, dropout, causal) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def children(self):
        return [self.tok_emb, self.emb_drop, *self.blocks, self.ln_f, self.head]

    def forward(self, tokens):
        """tokens has shape (B, T) with T <= max_len. Returns logits of shape (B, T, V)."""
        raise NotImplementedError("TODO: Step 10 in the project plan")

    def backward(self, dlogits):
        raise NotImplementedError("TODO: Step 10 in the project plan")

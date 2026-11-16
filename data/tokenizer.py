"""Character-level tokenizer (plan Step 12)."""

import numpy as np


class CharTokenizer:
    """Maps each distinct character in a training text to an integer id."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return np.array([self.stoi[ch] for ch in text], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

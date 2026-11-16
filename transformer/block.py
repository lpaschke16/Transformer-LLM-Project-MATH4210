"""Feed-forward network and the transformer block (plan Steps 7, 8 and 10).

Block (pre-LayerNorm)
    x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))
For y = x + F(x), the Jacobian is I + dF/dx, so in backward the upstream
gradient passes straight through the residual and is added to the branch.
"""

from transformer.attention import MultiHeadAttention
from transformer.layers import GELU, Dropout, Layer, LayerNorm, Linear


class FeedForward(Layer):
    """FFN(x) = W2 g(W1 x + b1) + b2 with hidden width 4C (Step 7)."""

    def __init__(self, d_model, rng, hidden_mult=4):
        super().__init__()
        self.fc1 = Linear(d_model, hidden_mult * d_model, rng)
        self.act = GELU()
        self.fc2 = Linear(hidden_mult * d_model, d_model, rng)

    def children(self):
        return [self.fc1, self.act, self.fc2]

    def forward(self, x):
        raise NotImplementedError("TODO: Step 7 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Step 7 in the project plan")


class Block(Layer):
    def __init__(self, d_model, n_heads, rng, dropout=0.0, causal=True):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rng, causal=causal)
        self.drop1 = Dropout(dropout, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, rng)
        self.drop2 = Dropout(dropout, rng)

    def children(self):
        return [self.ln1, self.attn, self.drop1, self.ln2, self.ffn, self.drop2]

    def forward(self, x):
        raise NotImplementedError("TODO: Step 10 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Step 10 in the project plan")

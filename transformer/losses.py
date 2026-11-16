"""Softmax and cross-entropy (plan Step 2)."""

import numpy as np


def softmax(z, axis=-1):
    """Numerically stable softmax. Subtract the max before exponentiating."""
    raise NotImplementedError("TODO: Step 2 in the project plan")


def softmax_cross_entropy(logits, targets):
    """Mean cross-entropy over every position, and its gradient.

    logits  has shape (..., V)
    targets has shape (...), integer class ids
    Returns (loss, dlogits) where dlogits = (p - one_hot(targets)) / N
    and N is the number of positions averaged over.
    Compute the loss with log-sum-exp so it never takes log(0).
    """
    raise NotImplementedError("TODO: Step 2 in the project plan")

# MATH 4210 Final Project: A Transformer from Scratch

A decoder-only transformer built in NumPy with every forward and backward pass derived by hand, then trained as a small character-level language model.

- **Part 1.** Build each layer, prove its gradient with a finite-difference check, assemble the model, and compare an encoder version against class methods on text classification.
- **Part 2.** Train the model on Tiny Shakespeare, compare it with a Bayesian n-gram baseline, and run controlled experiments.

The full step-by-step plan lives in the group's planning doc. Step numbers in the code match it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows uses .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Where things stand

`pytest` is the to-do list. Every unfinished function raises `NotImplementedError` with its step number, so a failing test tells you which step still needs doing.

| Step | What | File | Status |
| --- | --- | --- | --- |
| 0 | Gradient checker | `transformer/gradcheck.py` | Done |
| 1 | Linear layer | `transformer/layers.py` | Done (use it as the pattern) |
| 2 | Softmax and cross-entropy | `transformer/losses.py` | To do |
| 3 | Embedding and positional encoding | `transformer/layers.py` | To do |
| 4, 5 | Single-head and multi-head attention | `transformer/attention.py` | To do |
| 6 | LayerNorm | `transformer/layers.py` | To do |
| 7 | ReLU, GELU, feed-forward | `transformer/layers.py`, `transformer/block.py` | To do |
| 8 | Dropout (residuals live in the block) | `transformer/layers.py` | To do |
| 9 | AdamW, gradient clipping, schedule | `transformer/optim.py` | To do (SGD done) |
| 10 | Block and full model | `transformer/block.py`, `transformer/model.py` | To do |
| 11 | Classification loaders and baselines | `data/datasets.py`, `experiments/` | To do |
| 12, 13 | Tokenizer and batching | `data/tokenizer.py`, `data/datasets.py` | Done |
| 14 to 16 | n-gram baseline, training, experiments | `experiments/` | To do |

Run one file while you work on its step.

```bash
pytest tests/test_layers.py -k layernorm
```

## Layout

```
transformer/
  layers.py        Layer base class, Linear, Embedding, positions, LayerNorm, ReLU, GELU, Dropout
  attention.py     multi-head causal self-attention
  block.py         feed-forward network and one pre-LayerNorm block
  model.py         embeddings + N blocks + output head
  losses.py        softmax and cross-entropy
  optim.py         SGD, AdamW, gradient clipping, learning-rate schedule
  gradcheck.py     finite-difference gradient checker
data/
  tokenizer.py     character vocabulary
  datasets.py      Tiny Shakespeare loader, train/val split, batching
  raw/             downloaded data (not committed)
experiments/       one script per experiment, each writing a CSV to results/
tests/             one test file per step
```

## Conventions

- Activations have shape `(B, T, C)`, meaning batch, position, feature.
- Every layer subclasses `Layer`. `forward(x)` saves what backward needs. `backward(dout)` fills `self.grads` with the same keys and shapes as `self.params` and returns the gradient with respect to the input.
- Composite modules (attention, block, model) list their sub-layers in `children()`. `model.layers()` then returns every layer, which is what the optimizer, `train()`, `eval()` and the gradient checker use.
- Gradient checks run in float64. Training can use float32.
- Seed every random generator with `np.random.default_rng(seed)` and pass it in.
- Write the derivation for your step in the slides before or alongside the code.

## Gradient checking

`check_layer(layer, x, rng)` builds the loss `L = sum(R * layer(x))` with a fixed random `R`, so the upstream gradient is exactly `R`. It compares your analytic gradients with central differences on a random sample of entries and returns the worst relative error for the input and every parameter. Tests require errors below `1e-6`.

One thing the checker will teach you. The gradient of the key projection's bias is exactly zero, because adding the same vector to every key shifts a whole row of attention scores by a constant and softmax ignores that. `tests/test_attention.py` checks it.

## Data

Tiny Shakespeare (about 1.1M characters) comes from Andrej Karpathy's char-rnn repository.

```bash
curl -o data/raw/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Classification sets for Step 11 are IMDb, the UCI SMS Spam Collection, AG News, and 20 Newsgroups through scikit-learn.

## Workflow

- One branch per step, for example `step-06-layernorm`. Merge to `main` only when that step's tests pass.
- Credit any code adapted from an online source in a comment with the URL and date, as the syllabus requires.

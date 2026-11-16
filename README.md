# Transformer from Scratch and a Small LLM (MATH 4210)

We build a decoder-only (GPT-style) transformer in NumPy, derive every forward and backward pass by hand, prove each one with a finite-difference gradient check, then train the same code as a small character-level language model and run three or four controlled experiments.

Every step follows the same loop. Derive it on paper, code it, test it, and only then move on.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows uses .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

`pytest` doubles as the to-do list. Every unfinished function raises `NotImplementedError` naming its step, so a failing test tells you which step still needs doing. Run one file while you work.

```bash
pytest tests/test_layers.py -k layernorm
```

## Status

| Step | What | Code | Tests | Status |
| --- | --- | --- | --- | --- |
| 0 | Gradient checker | `transformer/gradcheck.py` | `test_gradcheck.py` | Done |
| 1 | Linear layer | `transformer/layers.py` | `test_layers.py` | Done (use it as the pattern) |
| 2 | Softmax and cross-entropy | `transformer/losses.py` | `test_losses.py` | To do |
| 3 | Embedding and positional encoding | `transformer/layers.py` | `test_layers.py` | To do |
| 4, 5 | Single-head and multi-head attention | `transformer/attention.py` | `test_attention.py` | To do |
| 6 | LayerNorm | `transformer/layers.py` | `test_layers.py` | To do |
| 7 | ReLU, GELU, feed-forward | `transformer/layers.py`, `transformer/block.py` | `test_layers.py`, `test_model.py` | To do |
| 8 | Dropout (residuals live in the block) | `transformer/layers.py` | `test_layers.py` | To do |
| 9 | AdamW, gradient clipping, schedule | `transformer/optim.py` | `test_optim.py` | To do (SGD done) |
| 10 | Block and full model | `transformer/block.py`, `transformer/model.py` | `test_model.py` | To do |
| 11 | Classification loaders and baselines | `data/datasets.py`, `experiments/` | | To do |
| 12, 13 | Tokenizer and batching | `data/tokenizer.py`, `data/datasets.py` | | Done |
| 14 to 16 | n-gram baseline, training, experiments | `experiments/` | | To do |

## Key decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Framework | NumPy only, with hand-written backward passes | Lecture 19 treats backpropagation as the chain rule written out; writing each backward pass yourself is the part that shows you understand the math |
| Architecture | Decoder-only, pre-LayerNorm, causal self-attention | Part 2 needs next-token prediction; switching the causal mask off turns the same blocks into an encoder for classification |
| Part 2 compute | Stay in NumPy at roughly 1M parameters | A model this size should train on a laptop CPU in tens of minutes to a couple of hours per run (estimate). For bigger sweeps, port the layers to PyTorch tensors and show autograd gradients match ours to 1e-6 |
| Tokens | Characters first, byte-pair encoding only as a stretch goal | A vocabulary of about 65 characters removes tokenizer bugs from the critical path |
| Scope | One working model plus 3 to 4 experiments done carefully | About three weeks remain before presentations |

## Repo layout

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
  tokenizer.py     character vocabulary (BPE later, optional)
  datasets.py      Tiny Shakespeare loader, train/val split, batching, classification loaders
  raw/             downloaded data (not committed)
experiments/       one script per experiment, each writing a CSV to results/
results/           experiment outputs
tests/             one test file per step
```

## Conventions

- Activations have shape `(B, T, C)`, meaning batch, sequence position, feature.
- Every layer subclasses `Layer`. `forward(x)` saves whatever `backward` will need. `backward(dout)` takes the upstream gradient, fills `self.grads` with the same keys and shapes as `self.params`, and returns the gradient with respect to the input. `Linear` in `transformer/layers.py` is the worked pattern.
- Composite modules (attention, block, model) list their sub-layers in `children()`. `model.layers()` then returns every layer, which is what the optimizer, `train()`, `eval()` and the gradient checker use.
- Gradient checks run in float64. Training can use float32.
- Seed every random generator with `np.random.default_rng(seed)` and pass it in, so results are reproducible.

## Gradient checking

For a scalar loss $L$ and each parameter entry, compare the analytic gradient with the central difference below. We use $h = 10^{-5}$ in float64 and require relative error under $10^{-6}$, checking a random sample of entries in each array to keep it fast.

$$
\frac{\partial L}{\partial \theta_j} \approx \frac{L(\theta + h e_j) - L(\theta - h e_j)}{2h}, \qquad \text{rel. error} = \frac{|g_{\text{analytic}} - g_{\text{numeric}}|}{\max(|g_{\text{analytic}}| + |g_{\text{numeric}}|,\ 10^{-3})}
$$

`check_layer(layer, x, rng)` tests a layer in isolation with the loss $L = \sum R \odot \text{layer}(x)$ for a fixed random tensor $R$. Its upstream gradient is just $R$, so any mismatch is the layer's fault.

The $10^{-3}$ floor matters because some true gradients are exactly zero. The key projection's bias is one, since adding the same vector to every key shifts a whole row of attention scores by a constant and softmax ignores that. Without the floor, round-off noise there would look like a 100% error. `tests/test_attention.py` checks that this gradient is zero.

## Part 1. Build the pieces (Steps 1 to 9)

Nine components, each derived, coded, and gradient-checked on its own before anything is wired together. Steps 1, 2 and 9 can start in parallel; Step 4 needs Steps 1 and 2.

### Step 1. Linear layer

**Derive.** For $Y = XW + b$ with $X$ of shape $N \times d_{in}$, every gradient has the shape of the thing it differentiates, and that fixes where the transposes go.

$$
\frac{\partial L}{\partial W} = X^{\top} \frac{\partial L}{\partial Y}, \qquad \frac{\partial L}{\partial b} = \mathbf{1}^{\top} \frac{\partial L}{\partial Y}, \qquad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^{\top}
$$

This is the matrix form of the $\partial R / \partial w$ formulas in Lecture 19 and ESL equations 11.12 to 11.15. For $(B, T, C)$ inputs, flatten to $(B \cdot T, C)$ first.

**Code.** `Linear` in `transformer/layers.py` (done). **Test.** Gradient check on $W$, $b$ and $X$.

### Step 2. Softmax and cross-entropy

**Derive.** With $p = \text{softmax}(z)$ and loss $L = -\log p_y$, the gradient collapses to a very simple form. This is the same softmax and cross-entropy as Lecture 19's classification network and multinomial logistic regression.

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad \frac{\partial L}{\partial z} = p - e_y
$$

Worked example. For $z = (2, 1, 0)$ and true class $y = 0$, $p = (0.665, 0.245, 0.090)$ and the gradient is $(-0.335, 0.245, 0.090)$. Do this by hand, then match it in code.

Attention needs the general softmax backward pass too, so derive the Jacobian and its vector-Jacobian product, where $g$ is the upstream gradient.

$$
J = \operatorname{diag}(p) - p\,p^{\top}, \qquad \frac{\partial L}{\partial z} = p \odot \left(g - \langle g, p \rangle \mathbf{1}\right)
$$

**Code.** `softmax` and `softmax_cross_entropy` in `transformer/losses.py`. Subtract the row max before exponentiating, and compute the loss with log-sum-exp. **Test.** Gradient check, plus the worked example. At random initialization the loss should be close to $\ln V$, which is about 4.17 for a 65-character vocabulary.

### Step 3. Token embedding and positional encoding

**Derive.** The input to the stack is $X = E[\text{tokens}] + P$. Looking up row $i$ of $E$ is a one-hot vector times $E$, which is exactly the embedding matrix in ISLP Section 10.5.1. Its gradient adds the upstream gradient into row $i$ once for every position where token $i$ appears. We use sinusoidal positions to start (no parameters).

$$
P_{t,2k} = \sin\left(t / 10000^{2k/C}\right), \qquad P_{t,2k+1} = \cos\left(t / 10000^{2k/C}\right)
$$

**Code.** `Embedding` and `sinusoidal_positions` in `transformer/layers.py`. Accumulate with `np.add.at(dE, tokens, dX)`. Writing `dE[tokens] += dX` silently drops repeated tokens. **Test.** Gradient check on a batch with repeated tokens, which is what catches that bug.

### Step 4. Single-head scaled dot-product attention (the core step)

**Derive the forward pass.** $M$ is the causal mask, 0 on and below the diagonal and $-\infty$ above it, so position $t$ only sees positions up to $t$.

$$
Q = XW_Q,\; K = XW_K,\; V = XW_V, \qquad S = \frac{QK^{\top}}{\sqrt{d_k}} + M, \qquad A = \operatorname{softmax}_{\text{rows}}(S), \qquad O = AV
$$

Why divide by $\sqrt{d_k}$. If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k$ is a sum of $d_k$ such products and has variance $d_k$. Dividing by $\sqrt{d_k}$ brings it back to variance 1, which keeps the softmax away from a near one-hot output where its gradient vanishes. Write this proof out for the presentation.

Worked example with 3 tokens and $d_k = 2$. Let $Q = K = V$ have rows $(1, 0)$, $(0, 1)$, $(1, 1)$. The causal attention weights are rows $(1, 0, 0)$, $(0.330, 0.670, 0)$ and $(0.248, 0.248, 0.503)$, and the outputs are $(1, 0)$, $(0.330, 0.670)$ and $(0.752, 0.752)$. Compute it by hand once; it is already a unit test.

**Derive the backward pass.** Given $dO$, apply Step 1 and Step 2 in reverse order.

$$
dV = A^{\top} dO, \quad dA = dO\,V^{\top}, \quad dS = A \odot \left(dA - \operatorname{rowsum}(dA \odot A)\right), \quad dQ = \frac{dS\,K}{\sqrt{d_k}}, \quad dK = \frac{dS^{\top} Q}{\sqrt{d_k}}
$$

Masked entries have $A = 0$, so they get zero gradient automatically. Then push $dQ$, $dK$ and $dV$ through the three projections with the Step 1 formulas, and sum the three contributions to $dX$.

**Code.** `MultiHeadAttention` in `transformer/attention.py`. **Test.** Gradient check. Causality check, where changing the input at position $t + 1$ must leave every output at positions up to $t$ exactly unchanged.

### Step 5. Multi-head attention

**Derive.** Split $C$ features into $h$ heads of size $d_k = C / h$ by reshaping $(B, T, C)$ into $(B, h, T, d_k)$. Run Step 4 in each head, concatenate, and apply an output projection $W_O$. Parameter count is $4C^2 + 4C$ with biases. The backward pass reverses the reshape and transpose (`split_heads` and `merge_heads` are provided).

**Test.** With $h = 1$ it must reproduce the Step 4 worked example. Gradient check with $h = 2$ and $h = 4$.

### Step 6. LayerNorm

**Derive.** Normalize each token's $C$ features, then rescale with learned $\gamma$ and $\beta$. This is the hardest derivation in the project, so do it together on a whiteboard. With $g = dy \odot \gamma$ and means taken over the $C$ features of one token, the result is below.

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \odot \hat{x} + \beta, \qquad dx = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left( g - \overline{g} - \hat{x} \odot \overline{g \odot \hat{x}} \right)
$$

The motivation connects to ESL Section 11.5.3 on scaling the inputs to a network.

**Code.** `LayerNorm` in `transformer/layers.py`. **Test.** Gradient check. With $\gamma = 1$ and $\beta = 0$, every token's output has mean 0 and variance 1.

### Step 7. Position-wise feed-forward network

**Derive.** $\text{FFN}(x) = W_2\, g(W_1 x + b_1) + b_2$ with a hidden width of $4C$, applied to each token separately. This is Lecture 19's single-hidden-layer network, so the backward pass reuses Step 1 plus $g'$. Use ReLU (as in lecture) or GELU.

**Code.** `ReLU` and `GELU` in `transformer/layers.py`, `FeedForward` in `transformer/block.py`. **Test.** Gradient check. With ReLU, inputs are nudged away from 0 so the finite difference does not straddle the kink.

### Step 8. Residual connections and dropout

**Derive.** For $y = x + F(x)$, the Jacobian is $I + \partial F / \partial x$. The identity term gives the gradient a path through the whole stack untouched, which is why deep stacks train when stacked sigmoid layers saturate. Dropout zeroes each activation with probability $p$ and divides survivors by $1 - p$, so the expected value is unchanged (ISLP Section 10.7.3).

**Code.** `Dropout` in `transformer/layers.py`; residuals go in `Block` (Step 10). **Test.** Dropout is off in evaluation mode. Averaged over many masks, the dropout output matches the input.

### Step 9. Optimizer and training loop

**Derive.** Start with minibatch SGD from Lecture 20. Then derive Adam, including why the bias correction divides by $1 - \beta^t$. AdamW adds decoupled weight decay, which shrinks weights the way the ridge penalty does in Lectures 3 and 4. Add linear warmup, cosine decay, and clipping of the global gradient norm at 1.0.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
$$

**Code.** `AdamW`, `clip_grad_norm` and `lr_schedule` in `transformer/optim.py` (`SGD` is done). **Test.** Each optimizer fits an ordinary least squares problem and must reach the normal-equation solution from Lecture 3.

## Step 10. Assemble the model and prove it works

The full model is a straight stack of the Part 1 pieces, and it counts as done only after it passes all five checks below.

1. Token embedding plus positional encoding (Step 3)
2. $N$ transformer blocks, each computing $x \leftarrow x + \text{MultiHead}(\text{LayerNorm}(x))$ and then $x \leftarrow x + \text{FFN}(\text{LayerNorm}(x))$ (Steps 5 to 8)
3. A final LayerNorm
4. A linear head from $C$ features to $V$ vocabulary logits, optionally sharing weights with the embedding matrix $E$
5. Softmax cross-entropy against the next token (Step 2)

The backward pass of the whole model is each block's backward pass called in reverse order. Each block has $12C^2 + 13C$ parameters (attention $4C^2 + 4C$, feed-forward $8C^2 + 5C$, two LayerNorms $4C$), and `test_model.py` asserts the code agrees.

**Code.** `Block` in `transformer/block.py`, `Transformer` in `transformer/model.py`.

### Verification ladder

- [ ] Full-model gradient check on a tiny config ($V = 7$, $C = 8$, $h = 2$, $N = 2$, $T = 5$) in float64
- [ ] Loss at initialization is close to $\ln V$
- [ ] Overfit a single batch of 4 sequences to near-zero loss. If this fails, there is a bug, and more data will not fix it
- [ ] Solve three simulated sequence tasks (copy a digit string, reverse it, sort it) to near-perfect accuracy
- [ ] Plot the attention maps on the reversal task and check whether the expected anti-diagonal pattern appears

The first three are in `test_model.py`. The simulated tasks double as our first experiment. Remove the positional encoding and rerun reversal. Without positions, self-attention treats its input as an unordered set, so the model should fail at reversal while still handling tasks that ignore order. That result is a clean slide explaining why positional encoding exists.

## Step 11. Text classification against class methods

Turn off the causal mask, average the block outputs over positions, and add a softmax classifier. That gives an encoder transformer we can run on real labeled data sets and compare head to head with methods from class.

| Data set | Type | Task | Why it fits |
| --- | --- | --- | --- |
| IMDb movie reviews | Real | Sentiment, 2 classes | ISLP Sections 10.4 and 10.5 use it, reporting about 88% test accuracy for lasso logistic regression on bag-of-words and 87% for an LSTM, so we get published reference points |
| SMS Spam Collection (UCI) | Real | Spam vs not spam | Short messages, fast to train in NumPy |
| AG News | Real | Topic, 4 classes | Multi-class test of the softmax head |
| 20 Newsgroups (subset of 4 to 5 groups) | Real | Topic | Built into scikit-learn, easy to load |
| Copy, reverse, sort digit strings | Simulated | Sequence to sequence | Already built in Step 10 |

Subsample and truncate (for example 5,000 training documents at 256 characters or tokens each) so NumPy training stays in minutes. Use the same subsample for every method so the comparison is fair.

Build TF-IDF or bag-of-words features, then fit these with scikit-learn (confirm with Dr. Short that this is fine for baselines).

- Lasso and ridge logistic regression (Lectures 4, 7 and 8; the ISLP Section 10.4 baseline)
- Naive Bayes (Lecture 9)
- LDA on PCA-reduced features (Lectures 9 and 21)
- Random forest (Lectures 14 to 16)
- Support vector machine (Lectures 17 and 18)
- K-nearest neighbors (Lecture 5)

Tune each with k-fold cross-validation (Lecture 11), hold out a test set nobody touches until the end, and report test accuracy in one table per data set.

Expect a small from-scratch transformer to lose to lasso logistic regression on some of these sets. That is a legitimate result and connects to Lecture 20's point that a simpler method might perform just as well, and to ISLP Section 10.6 on when deep learning pays off. Report it honestly and explain it with data size and model size.

## Part 2. Build a small LLM (Steps 12 to 16)

Train the Step 10 model on next-character prediction, beat a Bayesian n-gram baseline, and spend the remaining time on 3 or 4 experiments that each answer one question.

### Step 12. Corpus and tokenizer

We use Tiny Shakespeare (about 1.1M characters, 65 distinct characters) from Karpathy's char-rnn repo.

```bash
curl -o data/raw/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

`CharTokenizer` in `data/tokenizer.py` maps each character to an integer. `train_val_split` in `data/datasets.py` takes the first 90% for training and the last 10% for validation as contiguous blocks, since random splits leak overlapping windows between the two sets.

### Step 13. Batching

`get_batch` in `data/datasets.py` samples random windows of length $T + 1$ from the training stream. Inputs are the first $T$ characters and targets are the same window shifted by one, so a single forward pass gives $T$ predictions at once thanks to the causal mask.

### Step 14. Bayesian n-gram baseline

Count bigrams (then trigrams) and smooth with add-$\alpha$. The smoothed estimate is the posterior mean under a Dirichlet($\alpha$) prior, which is the many-sided version of the coin example from Lecture 5 and the Bayesian_Coin livescript.

$$
\hat{P}(c \mid b) = \frac{n_{bc} + \alpha}{n_b + \alpha V}, \qquad \text{perplexity} = \exp\left(-\frac{1}{N} \sum_{t=1}^{N} \log \hat{P}(x_t \mid x_{\lt t})\right)
$$

Report validation cross-entropy and perplexity for both this baseline and the transformer.

### Step 15. Train and sample

Starting config is $C = 128$, $N = 4$ blocks, $h = 4$ heads, $T = 128$, batch 32, AdamW with peak learning rate around 1e-3 after warmup. That is about 0.8M parameters ($4 \times (12C^2 + 13C)$ plus the $65 \times 128$ embedding). Log training and validation loss every few hundred steps and keep the checkpoint with the best validation loss (early stopping).

Generate text one character at a time by sampling from $\text{softmax}(z / \tau)$. Show samples at $\tau = 0.5$, 1.0 and 1.5 side by side, which is a good visual for explaining what temperature does to a probability distribution.

### Step 16. Experiments (pick 3 or 4)

| Question | What we vary | What we measure | Course tie-in |
| --- | --- | --- | --- |
| Does attention beat counting? | n-gram vs transformer | Validation perplexity | Bayesian inference, Lecture 5 |
| How does size trade off against fit? | C in {32, 64, 128} and N in {1, 2, 4} | Train and validation loss vs parameter count | Bias-variance tradeoff (Lecture 2), double descent (ISLP 10.8) |
| How much context helps? | T in {16, 32, 64, 128} | Validation loss | Model flexibility |
| Which parts matter? | Remove positions, residuals, or LayerNorm; 1 head vs 4 | Validation loss and training stability | Steps 3, 6, 8 derivations |
| Does regularization help at this scale? | Dropout rate and weight decay | Gap between train and validation loss | Ridge (Lecture 4), dropout (ISLP 10.7.3) |
| What did it learn? | Nothing, analysis only | Attention maps; PCA of learned character embeddings | PCA (Lecture 21) |

Run each setting with at least 2 random seeds and report the spread, since a single run can mislead. Stretch goals, only if time allows, are a byte-pair encoding tokenizer written from scratch and a PyTorch port of our layers checked against our NumPy gradients. See `experiments/README.md` for the planned scripts.

## Timeline and team split

The whole build fits in three weeks if Steps 1 to 9 run in parallel and the full-model gradient check passes by Nov 25.

| Workstream | Dates (2026) |
| --- | --- |
| Derivations and gradient checker | Nov 16 to Nov 19 |
| Build Steps 1 to 9 | Nov 17 to Nov 22 |
| Assemble and verify (Step 10) | Nov 21 to Nov 25 |
| **Gate. Full-model gradient check passes** | **Nov 25** |
| Data sets and class-method baselines | Nov 18 to Nov 29 |
| Transformer classifiers (Step 11) | Nov 24 to Nov 30 |
| LLM and experiments (Steps 12 to 16) | Nov 27 to Dec 4 |
| Slides and report | Dec 1 to Dec 6 |

Thanksgiving falls on Nov 26, so front-load Week 2. If the gate slips, cut Part 2 down to the n-gram comparison plus one experiment before cutting anything from Part 1. Confirm the presentation date on Canvas.

| Role | Owns | Steps |
| --- | --- | --- |
| Math lead | Written derivations for the slides, single-head and multi-head attention | 2, 4, 5 |
| Layers | Linear, embedding, LayerNorm, feed-forward, residuals and dropout | 1, 3, 6, 7, 8 |
| Training | Gradient checker, optimizer, training loop, the LLM runs | 0, 9, 12 to 16 |
| Data and baselines | Data loaders, scikit-learn baselines, classification runs, slide deck lead | 11, slides |

With three people, the Layers owner also takes the data loaders and everyone shares the baselines. Each person writes the derivation slide for the steps they own.

## Workflow

- One branch per step, for example `step-06-layernorm`. Merge to `main` only when that step's tests pass.
- Write the derivation for your step before or alongside the code.
- Credit any code adapted from an online source in a comment with the URL and date, as the syllabus requires.

## Pitfalls checklist

Most from-scratch transformer bugs fall into the eight traps below, and each one has a cheap test.

- **Softmax overflow.** Subtract the row max before exponentiating and compute the loss with log-sum-exp.
- **Mask applied after the softmax.** Add the mask to the scores before the softmax, never multiply the weights afterward. The causality test in Step 4 catches this.
- **Repeated-token embedding bug.** Use `np.add.at`, not fancy-index `+=` (Step 3).
- **Wrong transpose in a backward pass.** Check that every gradient has the same shape as its variable, then gradient-check.
- **Gradient checking in float32.** Finite differences need float64, or the errors look like bugs that are not there.
- **Bad initialization.** Weights drawn with standard deviation around 0.02 and zero biases keep early activations small, in line with ESL Section 11.5.1 on starting values near zero.
- **Learning rate too high.** If the loss spikes or turns into NaN, lower the peak rate, lengthen warmup, and confirm gradient clipping is on.
- **Evaluating with dropout on.** Every layer has a `training` flag. Call `model.eval()` for validation and sampling.

## References

Read the first two before Step 4. Everything else is consulted when its step comes up.

| Reference | Use it for |
| --- | --- |
| [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The original architecture, Steps 4, 5 and 10 |
| [Bishop and Bishop (2024), Deep Learning, Chapter 12 Transformers](https://www.bishopbook.com/) | Clean math notation for attention and transformer language models; Dr. Short's topic list already cites this book |
| ISLP Chapter 10 (Sections 10.4, 10.5.1, 10.6, 10.7, 10.8) | Embeddings, the IMDb baselines, backpropagation, dropout, double descent |
| ESL Sections 11.3 to 11.5 | Backpropagation equations 11.12 to 11.15, initialization, input scaling |
| MATH 4210 Lectures 19 and 20 | Course notation for networks, cross-entropy, gradient descent and SGD |
| [Ba, Kiros and Hinton (2016), Layer Normalization](https://arxiv.org/abs/1607.06450) | Step 6 |
| [Kingma and Ba (2014), Adam](https://arxiv.org/abs/1412.6980) and [Loshchilov and Hutter (2017), Decoupled Weight Decay](https://arxiv.org/abs/1711.05101) | Step 9 |
| [Hendrycks and Gimpel (2016), GELU](https://arxiv.org/abs/1606.08415) | Step 7 |
| [Karpathy, nanoGPT](https://github.com/karpathy/nanoGPT) | A PyTorch reference to compare shapes and loss curves against, and the Tiny Shakespeare setup. Credit anything borrowed, per the syllabus |

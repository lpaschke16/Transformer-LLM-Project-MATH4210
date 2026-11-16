# Experiments

One script per experiment (plan Steps 11 and 14 to 16). Each script

- fixes its random seeds and records them,
- writes its numbers to `results/<experiment>.csv`,
- saves any plots to `results/<experiment>.png`.

Planned scripts

| Script | Question |
| --- | --- |
| `toy_tasks.py` | Can the model copy, reverse and sort digit strings, and what breaks without positional encoding? |
| `classify.py` | Encoder transformer vs class methods on IMDb, SMS Spam, AG News, 20 Newsgroups |
| `ngram_baseline.py` | Bigram and trigram perplexity with add-alpha (Dirichlet) smoothing |
| `train_lm.py` | Train the character-level model on Tiny Shakespeare and sample from it |
| `size_sweep.py` | Train and validation loss vs parameter count |
| `ablations.py` | Remove positions, residuals or LayerNorm; 1 head vs 4 |

Run every setting with at least 2 seeds and report the spread.

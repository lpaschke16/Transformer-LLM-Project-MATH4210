"""Data loading and batching (plan Steps 11 to 13).

Raw files go in data/raw/, which git ignores. See the README for downloads.
"""

from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).parent / "raw"


def load_tiny_shakespeare(path=RAW_DIR / "tiny_shakespeare.txt"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with the command in the README."
        )
    return path.read_text(encoding="utf-8")


def train_val_split(ids, train_frac=0.9):
    """Contiguous split. A random split would leak overlapping windows."""
    n_train = int(len(ids) * train_frac)
    return ids[:n_train], ids[n_train:]


def get_batch(ids, batch_size, block_size, rng):
    """Random windows for next-token prediction (Step 13).

    x has shape (batch_size, block_size); y is x shifted left by one position.
    """
    starts = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[s:s + block_size] for s in starts])
    y = np.stack([ids[s + 1:s + block_size + 1] for s in starts])
    return x, y


# TODO (Step 11): loaders for the classification data sets.
# Each should return (train_texts, train_labels, test_texts, test_labels)
# so the transformer and the scikit-learn baselines see identical splits.
#   load_imdb()          IMDb reviews, 2 classes
#   load_sms_spam()      UCI SMS Spam Collection, 2 classes
#   load_ag_news()       AG News, 4 classes
#   load_newsgroups()    sklearn.datasets.fetch_20newsgroups, 4 to 5 groups

import random
from typing import List, Tuple
from pathlib import Path

def sort_data(data: List[Path], seed: int = 56) -> Tuple[List, List, List]:
    random.seed(seed)
    data = data.copy()  # avoid mutating the original list
    random.shuffle(data)

    n = len(data)

    # Require at least 3 files so each split gets one
    if n < 3:
        raise ValueError(
            f"sort_data requires at least 3 files, got {n}."
        )

    train_size = max(1, int(0.8 * n))
    test_size  = max(1, int(0.1 * n))

    # If train + test would exceed what's available, shrink train first
    if train_size + test_size >= n:
        test_size  = 1
        val_size   = 1
        train_size = n - test_size - val_size
    else:
        val_size = n - train_size - test_size

    train = data[:train_size]
    test  = data[train_size:train_size + test_size]
    val   = data[train_size + test_size:]

    print(f"train: {len(train)}, test: {len(test)}, val: {len(val)}")
    print(f"val files: {val}")
    return train, test, val

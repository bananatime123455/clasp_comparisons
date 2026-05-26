import random
from typing import List, Tuple
from pathlib import Path

def sort_data(data: List[Path], seed: int = 56) -> Tuple[List, List, List]:
    random.seed(seed)
    data = data.copy()  # avoid mutating the original list
    random.shuffle(data)

    train_size = int(0.8 * len(data))
    test_size  = int(0.1 * len(data))

    train = data[:train_size]
    test  = data[train_size:train_size + test_size]
    val   = data[train_size + test_size:]
    return train, test, val

def split_by_modality(training_data, testing_data):
    training_data_copy = training_data.copy()
    testing_data_copy = testing_data.copy()
    
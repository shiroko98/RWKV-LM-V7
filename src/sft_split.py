import math

import numpy as np


def compute_sft_tail_eval_count(
    document_count: int,
    tail_ratio: float = 0.0,
    tail_docs: int = 0,
    *,
    require_train_docs: bool = True,
) -> int:
    if document_count <= 0:
        raise ValueError("document_count must be positive.")
    if tail_docs < 0:
        raise ValueError("sft_eval_tail_docs must be non-negative.")
    if tail_ratio < 0 or tail_ratio > 1:
        raise ValueError("sft_eval_tail_ratio must be in [0, 1].")

    if tail_docs > 0:
        eval_count = tail_docs
    elif tail_ratio > 0:
        eval_count = max(1, int(math.ceil(document_count * tail_ratio)))
    else:
        eval_count = 0

    if require_train_docs and eval_count >= document_count:
        raise ValueError("SFT eval tail split would leave no training documents.")
    if eval_count > document_count:
        raise ValueError("SFT eval tail split cannot exceed total documents.")
    return eval_count


def compute_sft_train_document_count(document_count: int, tail_ratio: float = 0.0, tail_docs: int = 0) -> int:
    return document_count - compute_sft_tail_eval_count(document_count, tail_ratio, tail_docs)


def compute_sft_shuffled_split_indices(
    document_count: int,
    eval_count: int,
    *,
    eval_include_in_train: bool = False,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    if document_count <= 0:
        raise ValueError("document_count must be positive.")
    if eval_count < 0:
        raise ValueError("eval_count must be non-negative.")
    if eval_count > document_count:
        raise ValueError("eval_count cannot exceed total documents.")
    if not eval_include_in_train and eval_count >= document_count:
        raise ValueError("SFT eval tail split would leave no training documents.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    order = np.random.default_rng(seed).permutation(document_count)
    if eval_count > 0:
        eval_indices = order[-eval_count:]
        train_indices = order if eval_include_in_train else order[:-eval_count]
    else:
        eval_indices = np.array([], dtype=order.dtype)
        train_indices = order
    return train_indices.astype(np.int64, copy=False), eval_indices.astype(np.int64, copy=False)

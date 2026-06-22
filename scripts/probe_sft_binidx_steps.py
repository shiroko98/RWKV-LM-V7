#!/usr/bin/env python3
"""Probe SFT binidx sampling around optimizer steps without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - depends on invocation cwd.
    sys.path.insert(0, str(ROOT))

from src.binidx import MMapIndexedDataset, index_file_path  # noqa: E402
from src.sft_split import compute_sft_shuffled_split_indices, compute_sft_tail_eval_count  # noqa: E402


@dataclass(frozen=True)
class BatchConfig:
    num_nodes: int
    devices: int
    micro_bsz: int
    accumulate_grad_batches: int
    epoch_steps: int
    epoch_begin: int = 0
    sft_train_shuffle: int = 0
    sft_train_shuffle_seed: int = 1234

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.devices

    @property
    def real_bsz(self) -> int:
        return self.world_size * self.micro_bsz

    @property
    def effective_bsz(self) -> int:
        return self.real_bsz * self.accumulate_grad_batches

    @property
    def samples_per_epoch(self) -> int:
        return self.epoch_steps * self.effective_bsz


@dataclass(frozen=True)
class SplitInfo:
    total_documents: int
    train_documents: int
    eval_documents: int
    eval_include_in_train: int
    train_doc_indices: tuple[int, ...] | None = None
    eval_doc_indices: tuple[int, ...] | None = None

    def train_doc_index(self, offset: int) -> int:
        if self.train_doc_indices is not None:
            return self.train_doc_indices[int(offset)]
        return int(offset)


def normalize_prefix(path: str) -> str:
    if path.endswith(".idx") or path.endswith(".bin"):
        return path[:-4]
    return path


def count_documents(prefix: str) -> int:
    normalized = normalize_prefix(prefix)
    index_path = index_file_path(normalized)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing binidx index file: {index_path}")
    return len(MMapIndexedDataset.Index(index_path))


def compute_split_info(
    total_documents: int,
    *,
    eval_tail_ratio: float,
    eval_tail_docs: int,
    eval_include_in_train: int,
    sft_train_shuffle: int = 0,
    sft_train_shuffle_seed: int = 1234,
) -> SplitInfo:
    eval_documents = compute_sft_tail_eval_count(
        total_documents,
        eval_tail_ratio,
        eval_tail_docs,
        require_train_docs=not bool(eval_include_in_train),
    )
    train_documents = total_documents if eval_include_in_train else total_documents - eval_documents
    train_doc_indices = None
    eval_doc_indices = None
    if sft_train_shuffle and eval_documents > 0:
        train_indices, eval_indices = compute_sft_shuffled_split_indices(
            total_documents,
            eval_documents,
            eval_include_in_train=bool(eval_include_in_train),
            seed=sft_train_shuffle_seed,
        )
        eval_doc_indices = tuple(int(index) for index in eval_indices.tolist())
        if not eval_include_in_train:
            train_doc_indices = tuple(int(index) for index in train_indices.tolist())
    return SplitInfo(
        total_documents=total_documents,
        train_documents=train_documents,
        eval_documents=eval_documents,
        eval_include_in_train=eval_include_in_train,
        train_doc_indices=train_doc_indices,
        eval_doc_indices=eval_doc_indices,
    )


def auto_epoch_steps(train_documents: int, effective_bsz: int) -> int:
    if train_documents <= 0:
        raise ValueError("train_documents must be positive.")
    if effective_bsz <= 0:
        raise ValueError("effective_bsz must be positive.")
    return (train_documents + effective_bsz - 1) // effective_bsz


def parse_probe_steps(text: str) -> list[int]:
    steps: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(piece) for piece in part.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError(f"Invalid probe step range: {part!r}")
            start, stop = pieces[0], pieces[1]
            stride = pieces[2] if len(pieces) == 3 else 1
            if stride <= 0:
                raise ValueError("Range stride must be positive.")
            steps.extend(range(start, stop + 1, stride))
        elif "-" in part and not part.startswith("-"):
            start_s, stop_s = part.split("-", 1)
            start, stop = int(start_s), int(stop_s)
            steps.extend(range(start, stop + 1))
        else:
            steps.append(int(part))
    if not steps:
        raise ValueError("At least one probe step is required.")
    return steps


def logged_step_to_optimizer_index(step: int, step_base: str) -> int:
    if step_base == "zero":
        index = step
    elif step_base == "one":
        index = step - 1
    else:
        raise ValueError("step_base must be 'zero' or 'one'.")
    if index < 0:
        raise ValueError(f"Step {step} maps to a negative optimizer index.")
    return index


def sample_indices_for_optimizer_index(optimizer_index: int, config: BatchConfig) -> list[int]:
    if optimizer_index < 0:
        raise ValueError("optimizer_index must be non-negative.")
    if config.epoch_steps <= 0:
        raise ValueError("epoch_steps must be positive.")
    epoch = config.epoch_begin + optimizer_index // config.epoch_steps
    local_step = optimizer_index % config.epoch_steps
    base = epoch * config.samples_per_epoch + local_step * config.effective_bsz
    return [base + offset for offset in range(config.effective_bsz)]


def epoch_sample_offsets_for_optimizer_index(optimizer_index: int, config: BatchConfig) -> list[int]:
    if optimizer_index < 0:
        raise ValueError("optimizer_index must be non-negative.")
    if config.epoch_steps <= 0:
        raise ValueError("epoch_steps must be positive.")
    local_step = optimizer_index % config.epoch_steps
    base = local_step * config.effective_bsz
    return [base + offset for offset in range(config.effective_bsz)]


@lru_cache(maxsize=16)
def epoch_permutation(train_documents: int, seed: int, epoch: int) -> tuple[int, ...]:
    if train_documents <= 0:
        raise ValueError("train_documents must be positive.")
    rng = np.random.default_rng(seed + epoch)
    return tuple(int(index) for index in rng.permutation(train_documents))


def doc_indices_for_optimizer_index(optimizer_index: int, config: BatchConfig, split: SplitInfo) -> list[int]:
    if split.train_documents <= 0:
        raise ValueError("train split has no documents.")
    if not config.sft_train_shuffle:
        samples = sample_indices_for_optimizer_index(optimizer_index, config)
        offsets = [sample % split.train_documents for sample in samples]
        return [split.train_doc_index(offset) for offset in offsets]
    epoch = config.epoch_begin + optimizer_index // config.epoch_steps
    offsets = [sample % split.train_documents for sample in epoch_sample_offsets_for_optimizer_index(optimizer_index, config)]
    permutation = epoch_permutation(split.train_documents, config.sft_train_shuffle_seed, epoch)
    return [split.train_doc_index(permutation[offset]) for offset in offsets]


def window_doc_indices(
    logged_step: int,
    *,
    window_steps: int,
    step_base: str,
    config: BatchConfig,
    split: SplitInfo,
) -> list[int]:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive.")
    optimizer_start = logged_step_to_optimizer_index(logged_step, step_base)
    docs: list[int] = []
    for offset in range(window_steps):
        docs.extend(doc_indices_for_optimizer_index(optimizer_start + offset, config, split))
    return docs


def stable_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=8).hexdigest()


class SftBinidxProbe:
    def __init__(
        self,
        prefix: str,
        *,
        ctx_len: int,
        mask_prefix: str | None = None,
        pad_token_id: int = 65532,
        cache_size: int = 8192,
        vocab_path: str | None = None,
        show_text_chars: int = 0,
    ) -> None:
        self.prefix = normalize_prefix(prefix)
        self.ctx_len = ctx_len
        self.req_len = ctx_len + 1
        self.pad_token_id = pad_token_id
        self.tokens = MMapIndexedDataset(self.prefix)
        self.masks = MMapIndexedDataset(mask_prefix or f"{self.prefix}.mask")
        if len(self.tokens) != len(self.masks):
            raise ValueError(f"Token/mask document counts differ: {len(self.tokens)} != {len(self.masks)}")
        if not np.array_equal(self.tokens.sizes, self.masks.sizes):
            raise ValueError("Token/mask document sizes differ.")

        self.tokenizer = None
        self.show_text_chars = show_text_chars
        if vocab_path and show_text_chars > 0:
            from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

            self.tokenizer = TRIE_TOKENIZER(vocab_path)

        self._doc_stats = lru_cache(maxsize=cache_size)(self._uncached_doc_stats)

    def doc_stats(self, doc_index: int) -> dict:
        return dict(self._doc_stats(int(doc_index)))

    def _uncached_doc_stats(self, doc_index: int) -> tuple[tuple[str, object], ...]:
        token_ids = self.tokens[doc_index].astype(np.int64, copy=False)
        loss_mask = self.masks[doc_index].astype(np.int64, copy=False)
        raw_length = int(len(token_ids))
        if raw_length != len(loss_mask):
            raise ValueError(f"Document {doc_index} token/mask lengths differ: {raw_length} != {len(loss_mask)}")

        usable = min(raw_length, self.req_len)
        train_mask = loss_mask[1:usable]
        train_tokens = int(train_mask.sum())
        target_tokens = max(0, min(self.ctx_len, usable - 1))
        density = float(train_tokens / self.ctx_len) if self.ctx_len > 0 else 0.0
        tail_padding = count_tail_padding(token_ids[:usable], loss_mask[:usable], self.pad_token_id)

        result: dict[str, object] = {
            "doc_index": int(doc_index),
            "length": raw_length,
            "target_tokens": target_tokens,
            "train_tokens": train_tokens,
            "mask_density": density,
            "tail_padding": tail_padding,
            "token_hash": stable_hash(token_ids),
            "mask_hash": stable_hash(loss_mask),
            "too_long": raw_length > self.req_len,
        }
        if self.tokenizer is not None:
            text = decode_trainable_excerpt(self.tokenizer, token_ids[:usable], loss_mask[:usable], self.show_text_chars)
            result["trainable_excerpt"] = text
        return tuple(result.items())


def count_tail_padding(token_ids: np.ndarray, loss_mask: np.ndarray, pad_token_id: int) -> int:
    count = 0
    for token_id, mask in zip(reversed(token_ids.tolist()), reversed(loss_mask.tolist())):
        if int(token_id) == pad_token_id and int(mask) == 0:
            count += 1
        else:
            break
    return count


def decode_trainable_excerpt(tokenizer, token_ids: np.ndarray, loss_mask: np.ndarray, limit: int) -> str:
    selected = [int(tok) for tok, mask in zip(token_ids[1:].tolist(), loss_mask[1:].tolist()) if int(mask) == 1]
    if not selected:
        return ""
    try:
        text = tokenizer.decode(selected)
    except UnicodeDecodeError:
        text = tokenizer.decodeBytes(selected).decode("utf-8", errors="replace")
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit]


def summarize_window(logged_step: int, docs: list[int], stats: list[dict], *, step_base: str, window_steps: int) -> dict:
    unique_docs = len(set(docs))
    train_tokens = [int(item["train_tokens"]) for item in stats]
    densities = [float(item["mask_density"]) for item in stats]
    tail_padding = [int(item["tail_padding"]) for item in stats]
    token_hashes = [str(item["token_hash"]) for item in stats]
    duplicate_count = len(docs) - unique_docs
    digest = hashlib.blake2b("|".join(token_hashes).encode("utf-8"), digest_size=8).hexdigest()
    return {
        "kind": "window",
        "logged_step": logged_step,
        "step_base": step_base,
        "optimizer_start_index": logged_step_to_optimizer_index(logged_step, step_base),
        "window_steps": window_steps,
        "docs": len(docs),
        "unique_docs": unique_docs,
        "duplicate_docs": duplicate_count,
        "doc_min": min(docs) if docs else None,
        "doc_max": max(docs) if docs else None,
        "train_tokens_sum": int(sum(train_tokens)),
        "train_tokens_mean": float(np.mean(train_tokens)) if train_tokens else 0.0,
        "train_tokens_min": int(min(train_tokens)) if train_tokens else 0,
        "train_tokens_max": int(max(train_tokens)) if train_tokens else 0,
        "mask_density_mean": float(np.mean(densities)) if densities else 0.0,
        "mask_density_min": float(min(densities)) if densities else 0.0,
        "mask_density_max": float(max(densities)) if densities else 0.0,
        "tail_padding_sum": int(sum(tail_padding)),
        "too_long_docs": int(sum(1 for item in stats if item["too_long"])),
        "window_hash": digest,
    }


def compare_windows(left: dict, left_docs: list[int], right: dict, right_docs: list[int]) -> dict:
    left_set = set(left_docs)
    right_set = set(right_docs)
    common = len(left_set & right_set)
    union = len(left_set | right_set)
    ordered_equal = left_docs == right_docs
    return {
        "kind": "comparison",
        "left_step": left["logged_step"],
        "right_step": right["logged_step"],
        "common_docs": common,
        "left_docs": len(left_docs),
        "right_docs": len(right_docs),
        "jaccard": float(common / union) if union else 0.0,
        "ordered_equal": ordered_equal,
        "window_hash_equal": left["window_hash"] == right["window_hash"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", help="SFT binidx prefix, .bin, or .idx path.")
    parser.add_argument("--mask-file", default="", help="Optional mask prefix. Defaults to DATA_FILE + '.mask'.")
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--micro-bsz", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument(
        "--epoch-steps",
        type=int,
        default=0,
        help="Optimizer steps per epoch. 0 auto-computes one-pass ceil(train_docs/effective_bsz).",
    )
    parser.add_argument("--epoch-begin", type=int, default=0)
    parser.add_argument("--eval-tail-ratio", type=float, default=0.0)
    parser.add_argument("--eval-tail-docs", type=int, default=0)
    parser.add_argument("--eval-include-in-train", type=int, choices=(0, 1), default=0)
    parser.add_argument("--sft-train-shuffle", type=int, choices=(0, 1), default=0)
    parser.add_argument("--sft-train-shuffle-seed", type=int, default=1234)
    parser.add_argument(
        "--probe-steps",
        required=True,
        help="Logged/global optimizer steps to probe. Supports comma list and inclusive ranges like 3900:4300:50.",
    )
    parser.add_argument("--window-steps", type=int, default=1, help="Consecutive optimizer steps to include per probe.")
    parser.add_argument(
        "--step-base",
        choices=("one", "zero"),
        default="one",
        help="Use 'one' for wandb/save step numbers; use 'zero' for zero-based optimizer indices.",
    )
    parser.add_argument("--show-docs", type=int, default=8, help="Print this many per-doc rows for each window.")
    parser.add_argument("--pad-token-id", type=int, default=65532)
    parser.add_argument("--stat-cache-size", type=int, default=8192)
    parser.add_argument("--vocab-path", default="", help="Optional RWKV vocab for trainable text excerpts.")
    parser.add_argument("--show-text-chars", type=int, default=0)
    parser.add_argument("--jsonl-out", default="", help="Optional path to write machine-readable JSONL records.")
    return parser


def emit(record: dict, *, jsonl_file) -> None:
    text = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(text)
    if jsonl_file is not None:
        jsonl_file.write(text + "\n")


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("ctx_len", "num_nodes", "devices", "micro_bsz", "accumulate_grad_batches", "window_steps"):
        validate_positive(name.replace("_", "-"), int(getattr(args, name)))
    if args.sft_train_shuffle_seed < 0:
        raise ValueError("sft-train-shuffle-seed must be non-negative.")

    prefix = normalize_prefix(args.data_file)
    total_documents = count_documents(prefix)
    split = compute_split_info(
        total_documents,
        eval_tail_ratio=args.eval_tail_ratio,
        eval_tail_docs=args.eval_tail_docs,
        eval_include_in_train=args.eval_include_in_train,
        sft_train_shuffle=args.sft_train_shuffle,
        sft_train_shuffle_seed=args.sft_train_shuffle_seed,
    )
    provisional = BatchConfig(
        num_nodes=args.num_nodes,
        devices=args.devices,
        micro_bsz=args.micro_bsz,
        accumulate_grad_batches=args.accumulate_grad_batches,
        epoch_steps=1,
        epoch_begin=args.epoch_begin,
        sft_train_shuffle=args.sft_train_shuffle,
        sft_train_shuffle_seed=args.sft_train_shuffle_seed,
    )
    epoch_steps = args.epoch_steps or auto_epoch_steps(split.train_documents, provisional.effective_bsz)
    config = BatchConfig(
        num_nodes=args.num_nodes,
        devices=args.devices,
        micro_bsz=args.micro_bsz,
        accumulate_grad_batches=args.accumulate_grad_batches,
        epoch_steps=epoch_steps,
        epoch_begin=args.epoch_begin,
        sft_train_shuffle=args.sft_train_shuffle,
        sft_train_shuffle_seed=args.sft_train_shuffle_seed,
    )
    steps = parse_probe_steps(args.probe_steps)

    probe = SftBinidxProbe(
        prefix,
        ctx_len=args.ctx_len,
        mask_prefix=normalize_prefix(args.mask_file) if args.mask_file else None,
        pad_token_id=args.pad_token_id,
        cache_size=args.stat_cache_size,
        vocab_path=args.vocab_path or None,
        show_text_chars=args.show_text_chars,
    )

    jsonl_file = open(args.jsonl_out, "w", encoding="utf-8") if args.jsonl_out else None
    try:
        emit(
            {
                "kind": "config",
                "data_file": prefix,
                "total_documents": split.total_documents,
                "train_documents": split.train_documents,
                "eval_documents": split.eval_documents,
                "eval_include_in_train": split.eval_include_in_train,
                "num_nodes": config.num_nodes,
                "devices": config.devices,
                "world_size": config.world_size,
                "micro_bsz": config.micro_bsz,
                "accumulate_grad_batches": config.accumulate_grad_batches,
                "real_bsz": config.real_bsz,
                "effective_bsz": config.effective_bsz,
                "epoch_steps": config.epoch_steps,
                "samples_per_epoch": config.samples_per_epoch,
                "sft_train_shuffle": config.sft_train_shuffle,
                "sft_train_shuffle_seed": config.sft_train_shuffle_seed,
                "train_doc_indices_shuffled": split.train_doc_indices is not None,
                "eval_doc_indices_shuffled": split.eval_doc_indices is not None,
                "ctx_len": args.ctx_len,
                "step_base": args.step_base,
                "note": "No model is loaded; true loss/ppl cannot be computed from binidx alone.",
            },
            jsonl_file=jsonl_file,
        )

        window_records: list[tuple[dict, list[int]]] = []
        for step in steps:
            docs = window_doc_indices(
                step,
                window_steps=args.window_steps,
                step_base=args.step_base,
                config=config,
                split=split,
            )
            stats = [probe.doc_stats(doc_index) for doc_index in docs]
            summary = summarize_window(step, docs, stats, step_base=args.step_base, window_steps=args.window_steps)
            emit(summary, jsonl_file=jsonl_file)
            window_records.append((summary, docs))

            for rank, (doc_index, doc_stats) in enumerate(zip(docs[: args.show_docs], stats[: args.show_docs])):
                emit(
                    {
                        "kind": "doc",
                        "logged_step": step,
                        "position_in_window": rank,
                        **doc_stats,
                    },
                    jsonl_file=jsonl_file,
                )

        for i in range(len(window_records)):
            for j in range(i + 1, len(window_records)):
                left_summary, left_docs = window_records[i]
                right_summary, right_docs = window_records[j]
                emit(compare_windows(left_summary, left_docs, right_summary, right_docs), jsonl_file=jsonl_file)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - covered through main().
    raise SystemExit(main())

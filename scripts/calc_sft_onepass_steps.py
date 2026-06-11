#!/usr/bin/env python3
"""Calculate the one-pass SFT optimizer-step schedule from a binidx prefix."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.binidx import MMapIndexedDataset, index_file_path  # noqa: E402


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def _normalize_prefix(path: str) -> str:
    if path.endswith(".idx"):
        return path[:-4]
    if path.endswith(".bin"):
        return path[:-4]
    return path


def count_documents(prefix: str) -> int:
    normalized = _normalize_prefix(prefix)
    index_path = index_file_path(normalized)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing binidx index file: {index_path}")
    index = MMapIndexedDataset.Index(index_path)
    return len(index)


def compute_onepass_schedule(
    documents: int,
    *,
    num_nodes: int,
    devices: int,
    micro_bsz: int,
    accumulate_grad_batches: int,
    n_pass: int = 1,
    ctx_len: int | None = None,
) -> dict[str, int]:
    for name, value in {
        "documents": documents,
        "num_nodes": num_nodes,
        "devices": devices,
        "micro_bsz": micro_bsz,
        "accumulate_grad_batches": accumulate_grad_batches,
        "n_pass": n_pass,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    real_bsz = num_nodes * devices * micro_bsz
    effective_bsz = real_bsz * accumulate_grad_batches
    onepass_steps = math.ceil(documents / effective_bsz)
    samples_per_epoch = onepass_steps * effective_bsz
    result = {
        "documents": documents,
        "num_nodes": num_nodes,
        "devices": devices,
        "micro_bsz": micro_bsz,
        "real_bsz": real_bsz,
        "accumulate_grad_batches": accumulate_grad_batches,
        "effective_bsz": effective_bsz,
        "epoch_steps": onepass_steps,
        "epoch_count": n_pass,
        "total_optimizer_steps": onepass_steps * n_pass,
        "samples_per_epoch": samples_per_epoch,
        "extra_repeated_per_epoch": samples_per_epoch - documents,
    }
    if ctx_len is not None:
        if ctx_len <= 0:
            raise ValueError("ctx_len must be a positive integer.")
        result["ctx_len"] = ctx_len
        result["tokens_per_epoch"] = samples_per_epoch * ctx_len
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate EPOCH_STEPS / EPOCH_COUNT for one full pass over SFT binidx documents."
    )
    parser.add_argument("data_file", help="SFT binidx prefix. You may pass prefix, prefix.idx, or prefix.bin.")
    parser.add_argument("--num-nodes", type=int, default=_env_int(("N_NODE", "NUM_NODES"), 1))
    parser.add_argument("--devices", type=int, default=_env_int(("GPU_PER_NODE", "DEVICES"), 1))
    parser.add_argument("--micro-bsz", type=int, default=_env_int(("MICRO_BSZ",), 1))
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=_env_int(("ACCUMULATE_GRAD_BATCHES",), 1),
    )
    parser.add_argument("--n-pass", type=int, default=_env_int(("N_PASS",), 1))
    parser.add_argument("--ctx-len", type=int, default=None)
    parser.add_argument(
        "--save-hours",
        type=float,
        default=12.0,
        help="Wall-clock interval used for save_every_n_steps estimates.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        nargs="*",
        default=(35.0, 40.0, 45.0),
        help="One or more measured seconds/step values for checkpoint interval estimates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    documents = count_documents(args.data_file)
    schedule = compute_onepass_schedule(
        documents,
        num_nodes=args.num_nodes,
        devices=args.devices,
        micro_bsz=args.micro_bsz,
        accumulate_grad_batches=args.accumulate_grad_batches,
        n_pass=args.n_pass,
        ctx_len=args.ctx_len,
    )

    print(f"DATA_FILE={_normalize_prefix(args.data_file)}")
    for key in (
        "documents",
        "num_nodes",
        "devices",
        "micro_bsz",
        "real_bsz",
        "accumulate_grad_batches",
        "effective_bsz",
        "epoch_steps",
        "epoch_count",
        "total_optimizer_steps",
        "samples_per_epoch",
        "extra_repeated_per_epoch",
        "ctx_len",
        "tokens_per_epoch",
    ):
        if key in schedule:
            print(f"{key}={schedule[key]}")

    if args.save_hours > 0 and args.step_seconds:
        save_seconds = args.save_hours * 3600.0
        for seconds_per_step in args.step_seconds:
            if seconds_per_step <= 0:
                raise ValueError("All --step-seconds values must be positive.")
            steps = round(save_seconds / seconds_per_step)
            print(f"save_every_{args.save_hours:g}h_at_{seconds_per_step:g}s_per_step={steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

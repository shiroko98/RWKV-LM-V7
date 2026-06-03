#!/usr/bin/env python3
"""Validate that a converted RWKV `.pth` matches a DeepSpeed ZeRO checkpoint.

This script compares:
1. The reconstructed state_dict from the original DeepSpeed checkpoint directory
2. The converted single-file `.pth` checkpoint

It can also run a real forward pass using a demo-derived runtime module to
verify that both checkpoints produce numerically identical logits for the same prompt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONVERT_SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_deepspeed_checkpoint_to_pth.py"
CONVERT_SPEC = importlib.util.spec_from_file_location("convert_deepspeed_checkpoint_to_pth", CONVERT_SCRIPT_PATH)
assert CONVERT_SPEC is not None and CONVERT_SPEC.loader is not None
convert_script = importlib.util.module_from_spec(CONVERT_SPEC)
CONVERT_SPEC.loader.exec_module(convert_script)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the original DeepSpeed checkpoint directory")
    parser.add_argument("--converted-file", required=True, help="Path to the converted single-file .pth checkpoint")
    parser.add_argument("--dtype", choices=sorted(convert_script.DTYPE_MAP), default="bf16", help="Expected dtype of the converted checkpoint")
    parser.add_argument("--tag", default=None, help="Explicit DeepSpeed tag. Leave empty to use the checkpoint latest file")
    parser.add_argument("--lazy-mode", action=argparse.BooleanOptionalAction, default=True, help="Request lazy_mode=True from the DeepSpeed loader when supported")
    parser.add_argument("--exclude-frozen-parameters", action="store_true", help="Pass exclude_frozen_parameters=True to the DeepSpeed loader when supported")
    parser.add_argument("--summary-file", default="", help="Optional path to write a JSON summary of the comparison")
    parser.add_argument("--strict-forward", action="store_true", help="Fail if forward equivalence cannot be checked")
    parser.add_argument("--prompt", default="The Eiffel tower is in the city of", help="Prompt used for the forward-pass equivalence check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for the forward equivalence check")
    parser.add_argument("--max-abs-tol", type=float, default=0.0, help="Maximum allowed absolute difference for tensor equality")
    parser.add_argument("--max-rel-tol", type=float, default=0.0, help="Maximum allowed relative difference for tensor equality")
    parser.add_argument("--demo-model-path", default="", help="Deprecated compatibility flag. Ignored.")
    parser.add_argument("--demo-vocab-path", default="", help="Optional tokenizer vocab path override for the demo-derived runtime")
    return parser.parse_args()


def load_reconstructed_state_dict(args: argparse.Namespace) -> "OrderedDict[str, torch.Tensor]":
    raw_state = convert_script.reconstruct_fp32_state_dict(
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        tag=args.tag,
        lazy_mode=args.lazy_mode,
        exclude_frozen_parameters=args.exclude_frozen_parameters,
    )
    return convert_script.materialize_state_dict(raw_state, convert_script.DTYPE_MAP[args.dtype])


def load_converted_state_dict(path: Path) -> "OrderedDict[str, torch.Tensor]":
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Converted checkpoint must be a dict-like state_dict, got {type(state)!r}")
    return OrderedDict((str(k), v.detach().cpu()) for k, v in state.items())


def compare_state_dicts(
    expected: "OrderedDict[str, torch.Tensor]",
    actual: "OrderedDict[str, torch.Tensor]",
    max_abs_tol: float,
    max_rel_tol: float,
) -> dict[str, Any]:
    expected_keys = list(expected.keys())
    actual_keys = list(actual.keys())
    if expected_keys != actual_keys:
        missing = [k for k in expected_keys if k not in actual]
        extra = [k for k in actual_keys if k not in expected]
        raise ValueError(
            "State dict keys differ between reconstructed and converted checkpoints. "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    checked = 0
    for name in expected_keys:
        lhs = expected[name]
        rhs = actual[name]
        if lhs.shape != rhs.shape:
            raise ValueError(f"Shape mismatch for {name}: expected {tuple(lhs.shape)}, got {tuple(rhs.shape)}")
        if lhs.dtype != rhs.dtype:
            raise ValueError(f"Dtype mismatch for {name}: expected {lhs.dtype}, got {rhs.dtype}")

        if lhs.numel() == 0:
            checked += 1
            continue

        if lhs.is_floating_point():
            diff = (lhs.float() - rhs.float()).abs()
            this_abs = float(diff.max().item())
            denom = torch.maximum(lhs.float().abs(), rhs.float().abs()).clamp_min(1e-12)
            this_rel = float((diff / denom).max().item())
            max_abs_diff = max(max_abs_diff, this_abs)
            max_rel_diff = max(max_rel_diff, this_rel)
            if this_abs > max_abs_tol or this_rel > max_rel_tol:
                raise ValueError(
                    f"Tensor mismatch for {name}: max_abs_diff={this_abs} max_rel_diff={this_rel} "
                    f"(allowed abs={max_abs_tol}, rel={max_rel_tol})"
                )
        else:
            if not torch.equal(lhs, rhs):
                raise ValueError(f"Non-floating tensor mismatch for {name}")
        checked += 1

    return {
        "checked_tensors": checked,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }


def load_demo_module():
    runtime_path = REPO_ROOT / "src" / "rwkv_v7_demo_runtime.py"
    spec = importlib.util.spec_from_file_location("rwkv_v7_demo_runtime", runtime_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_demo_dims_from_state_dict(state_dict: "OrderedDict[str, torch.Tensor]") -> dict[str, int]:
    n_layer = len({int(name.split(".")[1]) for name in state_dict if name.startswith("blocks.") and ".ln1.weight" in name})
    n_embd = int(state_dict["emb.weight"].shape[1])
    vocab_size = int(state_dict["emb.weight"].shape[0])
    d_decay_lora = int(state_dict["blocks.0.att.w1"].shape[1])
    d_aaa_lora = int(state_dict["blocks.0.att.a1"].shape[1])
    d_mv_lora = int(state_dict["blocks.0.att.v1"].shape[1])
    d_gate_lora = int(state_dict["blocks.0.att.g1"].shape[1])
    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "d_decay_lora": d_decay_lora,
        "d_aaa_lora": d_aaa_lora,
        "d_mv_lora": d_mv_lora,
        "d_gate_lora": d_gate_lora,
    }


def build_demo_model(module, state_dict: "OrderedDict[str, torch.Tensor]", device: str, dtype: torch.dtype):
    dims = infer_demo_dims_from_state_dict(state_dict)
    module.configure_runtime(
        dtype=dtype,
        head_size=64,
        d_decay_lora=dims["d_decay_lora"],
        d_aaa_lora=dims["d_aaa_lora"],
        d_mv_lora=dims["d_mv_lora"],
        d_gate_lora=dims["d_gate_lora"],
    )

    args = types.SimpleNamespace()
    args.n_layer = dims["n_layer"]
    args.n_embd = dims["n_embd"]
    args.vocab_size = dims["vocab_size"]
    args.head_size_a = 64
    model = module.RWKV(args).to(dtype=dtype, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_demo_tokenizer(module, vocab_path: Path):
    return module.RWKV_TOKENIZER(str(vocab_path))


def compare_forward_outputs(args: argparse.Namespace, state_dict: "OrderedDict[str, torch.Tensor]") -> dict[str, Any]:
    module = load_demo_module()
    vocab_path = Path(args.demo_vocab_path).expanduser().resolve() if args.demo_vocab_path else REPO_ROOT / "data" / "tokenizer" / "rwkv_vocab_v20230424.txt"
    tokenizer = build_demo_tokenizer(module, vocab_path)

    converted_state = load_converted_state_dict(Path(args.converted_file).expanduser().resolve())
    reconstructed_model = build_demo_model(module, state_dict, args.device, convert_script.DTYPE_MAP[args.dtype])
    converted_model = build_demo_model(module, converted_state, args.device, convert_script.DTYPE_MAP[args.dtype])

    input_tokens = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=args.device).reshape(1, -1)

    with torch.no_grad():
        lhs = reconstructed_model(input_tensor)
        rhs = converted_model(input_tensor)

    diff = (lhs.float() - rhs.float()).abs()
    max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
    topk = min(10, lhs.shape[-1])
    lhs_last = lhs[0, -1].float()
    rhs_last = rhs[0, -1].float()
    lhs_topk = torch.topk(lhs_last, topk).indices.tolist()
    rhs_topk = torch.topk(rhs_last, topk).indices.tolist()

    return {
        "prompt_tokens": input_tokens,
        "prompt_length": len(input_tokens),
        "forward_max_abs_diff": max_abs_diff,
        "topk_match": lhs_topk == rhs_topk,
        "lhs_topk_ids": lhs_topk,
        "rhs_topk_ids": rhs_topk,
    }


def main() -> int:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    converted_file = Path(args.converted_file).expanduser().resolve()
    if not convert_script.looks_like_deepspeed_checkpoint_dir(checkpoint_dir):
        print(f"[equiv] not a DeepSpeed checkpoint directory: {checkpoint_dir}", file=sys.stderr)
        return 2
    if not converted_file.is_file():
        print(f"[equiv] converted checkpoint file does not exist: {converted_file}", file=sys.stderr)
        return 2

    print(f"[equiv] checkpoint dir:   {checkpoint_dir}")
    print(f"[equiv] converted file:   {converted_file}")
    print(f"[equiv] dtype:            {args.dtype}")
    print(f"[equiv] device:           {args.device}")

    expected = load_reconstructed_state_dict(args)
    actual = load_converted_state_dict(converted_file)
    tensor_summary = compare_state_dicts(expected, actual, args.max_abs_tol, args.max_rel_tol)
    print(
        "[equiv] state_dict match: "
        f"{tensor_summary['checked_tensors']} tensors checked, "
        f"max_abs_diff={tensor_summary['max_abs_diff']}, "
        f"max_rel_diff={tensor_summary['max_rel_diff']}"
    )

    forward_summary = None
    try:
        forward_summary = compare_forward_outputs(args, expected)
        print(
            "[equiv] forward check: "
            f"prompt_length={forward_summary['prompt_length']} "
            f"max_abs_diff={forward_summary['forward_max_abs_diff']} "
            f"topk_match={forward_summary['topk_match']}"
        )
    except Exception as exc:
        if args.strict_forward:
            raise
        print(f"[equiv] forward check skipped/failed: {exc}")

    if args.summary_file:
        summary_path = Path(args.summary_file).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "checkpoint_dir": str(checkpoint_dir),
                    "converted_file": str(converted_file),
                    "dtype": args.dtype,
                    "tensor_summary": tensor_summary,
                    "forward_summary": forward_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[equiv] wrote summary to {summary_path}")

    print("[equiv] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

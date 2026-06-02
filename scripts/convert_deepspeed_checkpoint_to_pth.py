#!/usr/bin/env python3
"""Convert a DeepSpeed ZeRO checkpoint directory into a single RWKV .pth file.

This script reconstructs a full state_dict from a ZeRO-2/3 checkpoint directory and
optionally casts it to bf16/fp16/fp32 before saving as a plain PyTorch .pth file.

It also supports writing a parameter summary text file in the same format as:
  Parameter: emb.weight, Shape: torch.Size([65536, 4096]), Dtype: torch.bfloat16
  ...
  Total parameters: 13270298624
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch

SUMMARY_LINE_RE = re.compile(
    r"^Parameter: (?P<name>.+), Shape: torch\.Size\((?P<shape>\[[^\]]*\])\), Dtype: (?P<dtype>torch\.[A-Za-z0-9_]+)$"
)
TOTAL_LINE_RE = re.compile(r"^Total parameters: (?P<total>\d+)$")
DEFAULT_STRIP_PREFIXES = ("_forward_module.",)
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def looks_like_deepspeed_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        entries = {entry.name for entry in path.iterdir()}
    except OSError:
        return False
    if "latest" in entries:
        return True
    return any(
        name.endswith("_model_states.pt") or name.endswith("_optim_states.pt") or name == "zero_to_fp32.py"
        for name in entries
    )


def normalize_param_name(name: str, strip_prefixes: Iterable[str] = DEFAULT_STRIP_PREFIXES) -> str:
    for prefix in strip_prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def materialize_zero_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    contiguous = getattr(obj, "contiguous", None)
    if callable(contiguous):
        tensor = contiguous()
        if isinstance(tensor, torch.Tensor):
            return tensor
    raise TypeError(f"Unsupported parameter payload type from DeepSpeed export: {type(obj)!r}")


def cast_tensor_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        return tensor.to(dtype=target_dtype)
    return tensor


def build_summary_lines(state_dict: "OrderedDict[str, torch.Tensor]") -> list[str]:
    lines: list[str] = []
    total_params = 0
    for name, tensor in state_dict.items():
        lines.append(f"Parameter: {name}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        total_params += tensor.numel()
    lines.append(f"Total parameters: {total_params}")
    return lines


def parse_summary_lines(lines: Iterable[str]):
    params = []
    total_params = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        match = SUMMARY_LINE_RE.match(line)
        if match:
            shape_text = match.group("shape").strip()[1:-1].strip()
            shape = tuple(int(part.strip()) for part in shape_text.split(",")) if shape_text else tuple()
            params.append((match.group("name"), shape, match.group("dtype")))
            continue
        match = TOTAL_LINE_RE.match(line)
        if match:
            total_params = int(match.group("total"))
    return params, total_params


def verify_summary(reference_path: Path, state_dict: "OrderedDict[str, torch.Tensor]") -> None:
    params, total_params = parse_summary_lines(reference_path.read_text(encoding="utf-8", errors="replace").splitlines())
    actual_lines = build_summary_lines(state_dict)
    actual_params, actual_total = parse_summary_lines(actual_lines)

    if len(params) != len(actual_params):
        raise ValueError(
            f"Summary parameter count mismatch: reference has {len(params)} entries, converted model has {len(actual_params)}."
        )

    for idx, (expected, actual) in enumerate(zip(params, actual_params)):
        if expected != actual:
            raise ValueError(
                "Summary mismatch at parameter index "
                f"{idx}: expected {expected}, got {actual}."
            )

    if total_params is not None and total_params != actual_total:
        raise ValueError(
            f"Total parameter mismatch: reference has {total_params}, converted model has {actual_total}."
        )


def load_zero_to_fp32_module(checkpoint_dir: Path):
    local_script = checkpoint_dir / "zero_to_fp32.py"
    if local_script.exists():
        spec = importlib.util.spec_from_file_location("rwkv_zero_to_fp32", local_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load local zero_to_fp32.py from {local_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        from deepspeed.utils import zero_to_fp32 as module
    except ImportError as exc:
        raise ImportError(
            "Could not import DeepSpeed zero_to_fp32 helpers and no local zero_to_fp32.py was found in the checkpoint directory."
        ) from exc
    return module


def reconstruct_fp32_state_dict(
    checkpoint_dir: Path,
    tag: str | None,
    lazy_mode: bool,
    exclude_frozen_parameters: bool,
):
    module = load_zero_to_fp32_module(checkpoint_dir)
    fn = getattr(module, "get_fp32_state_dict_from_zero_checkpoint", None)
    if fn is None:
        raise AttributeError("zero_to_fp32 helper does not expose get_fp32_state_dict_from_zero_checkpoint")

    kwargs = {}
    signature = inspect.signature(fn)
    if "tag" in signature.parameters and tag is not None:
        kwargs["tag"] = tag
    if "exclude_frozen_parameters" in signature.parameters:
        kwargs["exclude_frozen_parameters"] = exclude_frozen_parameters
    if "lazy_mode" in signature.parameters:
        kwargs["lazy_mode"] = lazy_mode
    return fn(str(checkpoint_dir), **kwargs)


def materialize_state_dict(
    state_dict,
    target_dtype: torch.dtype,
    strip_prefixes: Iterable[str] = DEFAULT_STRIP_PREFIXES,
) -> "OrderedDict[str, torch.Tensor]":
    converted: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    total = len(state_dict)
    for idx, (name, value) in enumerate(state_dict.items(), start=1):
        tensor = materialize_zero_tensor(value)
        tensor = cast_tensor_dtype(tensor, target_dtype)
        converted[normalize_param_name(name, strip_prefixes)] = tensor
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"[convert-zero] materialized {idx}/{total}: {name} -> {tensor.dtype} {tuple(tensor.shape)}")
    return converted


def default_output_path(checkpoint_dir: Path, dtype_name: str) -> Path:
    return checkpoint_dir.parent / f"{checkpoint_dir.stem}.{dtype_name}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the DeepSpeed checkpoint directory, e.g. /path/to/rwkv-step-200.pth")
    parser.add_argument("--output-file", default="", help="Output .pth path. Defaults to a sibling file like rwkv-step-200.bf16.pth")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="bf16", help="Floating-point dtype for the saved .pth file")
    parser.add_argument("--tag", default=None, help="Explicit DeepSpeed tag. Leave empty to use the tag from the checkpoint's latest file")
    parser.add_argument("--summary-file", default="", help="Optional path to write a parameter summary text file")
    parser.add_argument("--verify-summary-file", default="", help="Optional reference summary text file to compare against after conversion")
    parser.add_argument("--exclude-frozen-parameters", action="store_true", help="Pass exclude_frozen_parameters=True to the DeepSpeed loader when supported")
    parser.add_argument("--lazy-mode", action=argparse.BooleanOptionalAction, default=True, help="Request lazy_mode=True from the DeepSpeed loader when supported")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not looks_like_deepspeed_checkpoint_dir(checkpoint_dir):
        print(f"[convert-zero] not a DeepSpeed checkpoint directory: {checkpoint_dir}", file=sys.stderr)
        return 2

    output_file = Path(args.output_file).expanduser().resolve() if args.output_file else default_output_path(checkpoint_dir, args.dtype)
    summary_file = Path(args.summary_file).expanduser().resolve() if args.summary_file else None
    verify_summary_file = Path(args.verify_summary_file).expanduser().resolve() if args.verify_summary_file else None

    print(f"[convert-zero] checkpoint dir: {checkpoint_dir}")
    print(f"[convert-zero] output file:    {output_file}")
    print(f"[convert-zero] target dtype:   {args.dtype}")
    if summary_file is not None:
        print(f"[convert-zero] summary file:   {summary_file}")
    if verify_summary_file is not None:
        print(f"[convert-zero] verify file:    {verify_summary_file}")

    raw_state_dict = reconstruct_fp32_state_dict(
        checkpoint_dir=checkpoint_dir,
        tag=args.tag,
        lazy_mode=args.lazy_mode,
        exclude_frozen_parameters=args.exclude_frozen_parameters,
    )
    state_dict = materialize_state_dict(raw_state_dict, DTYPE_MAP[args.dtype])

    if verify_summary_file is not None:
        verify_summary(verify_summary_file, state_dict)
        print("[convert-zero] summary verification passed")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_file)
    print(f"[convert-zero] wrote consolidated checkpoint to {output_file}")

    if summary_file is not None:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text("\n".join(build_summary_lines(state_dict)) + "\n", encoding="utf-8")
        print(f"[convert-zero] wrote summary to {summary_file}")

    print(f"[convert-zero] total parameters: {sum(t.numel() for t in state_dict.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

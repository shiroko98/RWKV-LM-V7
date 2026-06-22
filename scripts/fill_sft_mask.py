#!/usr/bin/env python3
"""Fill an SFT binidx mask sidecar with a constant value."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binidx import MMapIndexedDataset, data_file_path, index_file_path  # noqa: E402


def resolve_mask_prefix(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    text = str(raw)
    if text.endswith(".mask.bin"):
        return Path(text[: -len(".bin")])
    if text.endswith(".mask.idx"):
        return Path(text[: -len(".idx")])
    if text.endswith(".mask"):
        return raw
    return Path(text + ".mask")


def coerce_fill_value(value: int, dtype: np.dtype) -> object:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        int_value = int(value)
        info = np.iinfo(dtype)
        if int_value < info.min or int_value > info.max:
            raise ValueError(f"value {value!r} cannot be safely stored as {dtype}")
        return dtype.type(int_value).item()
    if np.issubdtype(dtype, np.floating):
        return dtype.type(value).item()
    raise ValueError(f"unsupported mask dtype {dtype}")


def fill_mask_bin(
    mask_prefix: str | Path,
    *,
    output_prefix: str | Path | None = None,
    value: int = 1,
    in_place: bool = False,
    chunk_elements: int = 256 * 1024 * 1024,
) -> dict[str, object]:
    mask_prefix = resolve_mask_prefix(mask_prefix)
    input_bin = Path(data_file_path(str(mask_prefix)))
    input_idx = Path(index_file_path(str(mask_prefix)))
    if not input_bin.is_file():
        raise FileNotFoundError(f"missing mask bin file: {input_bin}")
    if not input_idx.is_file():
        raise FileNotFoundError(f"missing mask idx file: {input_idx}")
    if in_place and output_prefix is not None:
        raise ValueError("--output-prefix cannot be used with --in-place")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")

    index = MMapIndexedDataset.Index(str(input_idx))
    dtype = np.dtype(index.dtype)
    element_count = int(input_bin.stat().st_size // dtype.itemsize)
    expected_elements = int(np.asarray(index.sizes, dtype=np.int64).sum())
    expected_bytes = expected_elements * dtype.itemsize
    actual_bytes = input_bin.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"mask bin size mismatch: {input_bin} has {actual_bytes} bytes, "
            f"but idx expects {expected_bytes} bytes"
        )
    fill_value = coerce_fill_value(value, dtype)

    output_prefix_path = mask_prefix if in_place else Path(output_prefix).expanduser() if output_prefix else mask_prefix.with_name(mask_prefix.name + ".all1")
    output_bin = Path(data_file_path(str(output_prefix_path)))
    output_idx = Path(index_file_path(str(output_prefix_path)))
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    if in_place:
        target = np.memmap(input_bin, dtype=dtype, mode="r+", shape=(element_count,))
    else:
        shutil.copy2(input_idx, output_idx)
        with open(output_bin, "wb") as stream:
            stream.truncate(actual_bytes)
        target = np.memmap(output_bin, dtype=dtype, mode="r+", shape=(element_count,))

    try:
        for start in range(0, element_count, chunk_elements):
            stop = min(start + chunk_elements, element_count)
            target[start:stop] = fill_value
        target.flush()
    finally:
        del target

    return {
        "input_prefix": str(mask_prefix),
        "output_prefix": str(output_prefix_path),
        "bin_path": str(input_bin if in_place else output_bin),
        "idx_path": str(input_idx if in_place else output_idx),
        "dtype": str(dtype),
        "documents": int(len(index)),
        "elements": element_count,
        "value": value,
        "in_place": in_place,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help=(
            "SFT dataset prefix, mask prefix, or mask bin path. Examples: "
            "SFT_RWKV7_13B, SFT_RWKV7_13B.mask, SFT_RWKV7_13B.mask.bin"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output mask prefix. Default: <input-mask-prefix>.all1. Mutually exclusive with --in-place.",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input .mask.bin in place.")
    parser.add_argument("--value", type=int, default=1, help="Mask value to write everywhere. Default: 1.")
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=256 * 1024 * 1024,
        help="Number of mask elements to fill per mmap slice.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = fill_mask_bin(
        resolve_mask_prefix(args.path),
        output_prefix=args.output_prefix or None,
        value=args.value,
        in_place=args.in_place,
        chunk_elements=args.chunk_elements,
    )
    print("### SFT mask fill complete")
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

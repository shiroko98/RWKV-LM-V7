import argparse
import fileinput
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402
except ModuleNotFoundError:
    from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

from src.binidx import MMapIndexedDataset  # noqa: E402

"""
How to use:

python make_data.py demo.jsonl 3 4096

This will:
==> shuffle & duplicate demo.jsonl (for 3 epochs, good for finetuning)
note: this will be very slow for large jsonl and we need more efficient code.
==> load jsonl and tokenize
==> save as demo.bin & demo.idx
==> compute "magic_prime" for ctxlen 4096

Example:

Assume your source jsonl is:
{"text":"aa"}
{"text":"bb"}
{"text":"cc"}
{"text":"dd"}

The final binidx will be like (here "/" means end_of_doc, which is actually token [0]):
bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/

where the data is repeated 3 times (each time with different shuffle)
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = REPO_ROOT / "rwkv_vocab_v20260603.txt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw JSONL text into an RWKV binidx dataset."
    )
    parser.add_argument("input_jsonl", type=str)
    parser.add_argument("n_epoch", type=int)
    parser.add_argument("ctx_len", type=int)
    parser.add_argument(
        "--vocab",
        type=str,
        default=str(DEFAULT_VOCAB),
        help=(
            "RWKV vocab file. Defaults to the special-first 20260603 vocab; "
            "pass an older vocab explicitly only for legacy compatibility."
        ),
    )
    return parser


class MMapIndexedDatasetBuilder:
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


def add_raw(raw, *, tokenizer: TRIE_TOKENIZER, builder, count: int) -> int:
    out = tokenizer.encode(raw)
    if tokenizer.decode(out) != raw:
        raise ValueError("Tokenizer failed the raw-text round-trip check.")
    out.append(0)  # [0] = end_of_doc for rwkv tokenizer
    builder.add_item(np.array(out, dtype=np.uint16))
    builder.end_document()
    if count % 500 == 0:
        print(count, end=" ", flush=True)
    return count + 1


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.n_epoch <= 0:
        raise ValueError("n_epoch must be a positive integer.")
    if args.ctx_len <= 0:
        raise ValueError("ctx_len must be a positive integer.")

    n_epoch = args.n_epoch
    input_file = args.input_jsonl
    out_name = os.path.splitext(os.path.basename(input_file))[0]
    ctx_len = args.ctx_len
    temp_file = "make_data_temp.jsonl"
    tokenizer = TRIE_TOKENIZER(args.vocab, strict_length=True)

    print(f"### Vocab: {Path(args.vocab).resolve()}")
    print(f"### Convert {input_file} to {out_name}.bin/idx...")

    with open(input_file, "r", encoding="utf-8") as file:
        non_empty_lines = [line.strip() for line in file if line.strip()]

    print(f"### Found {len(non_empty_lines)} non-empty lines in {input_file}")

    with open(temp_file, "w", encoding="utf-8") as file:
        for epoch in range(n_epoch):
            print(f"Shuffle: {epoch + 1} out of {n_epoch}")
            random.shuffle(non_empty_lines)
            for entry in non_empty_lines:
                file.write(entry + "\n")

    print("### Building binidx...")
    builder = MMapIndexedDatasetBuilder(f"{out_name}.bin")
    count = 0
    with fileinput.input(temp_file, encoding="utf-8") as input_lines:
        for line in input_lines:
            count = add_raw(
                json.loads(line)["text"],
                tokenizer=tokenizer,
                builder=builder,
                count=count,
            )
    builder.finalize(f"{out_name}.idx")
    print("done")

    print("### Verifying result...")
    data = MMapIndexedDataset(out_name)
    data_len = len(data)
    data_size = len(data._bin_buffer) // data._index._dtype_size

    preview_limit = 100
    for idx in [0, data_len - 1]:
        _, size = data._index[idx]
        token_ids = data.get(idx=idx, offset=0, length=size).astype(int)
        print("-" * 70 + f"[{out_name} idx {idx} sz {size}]")
        assert token_ids[-1] == 0
        token_ids = token_ids[:-1]
        if len(token_ids) > preview_limit:
            print(tokenizer.decode(token_ids[:preview_limit]))
            print("· " * 30)
            print(tokenizer.decode(token_ids[-preview_limit:]))
        else:
            print(tokenizer.decode(token_ids))

    print(
        f"{'-' * 80}\n### Final {out_name}.bin/idx has {data_size} tokens, "
        f"{data_len} items. Dtype {data._index.dtype}"
    )

    if data_size >= ctx_len * 3:
        n_chunk = int(data_size // ctx_len) - 1
        for candidate in range(n_chunk, 0, -1):
            if candidate % 3 == 2 and is_prime(candidate):
                print(f"\n### magic_prime = {candidate} (for ctxlen {ctx_len})")
                print(
                    f"\n--my_exit_tokens {data_size} --magic_prime {candidate} "
                    f"--ctx_len {ctx_len}\n"
                )
                return


if __name__ == "__main__":
    main()

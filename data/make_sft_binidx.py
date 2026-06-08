from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sft_binidx import build_binidx_dataset  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render chat-template SFT JSONL into binidx tokens plus a loss-mask sidecar dataset."
    )
    parser.add_argument("input_jsonl", type=str, nargs="+")
    parser.add_argument("--out-prefix", "--output-prefix", dest="out_prefix", type=str, default=None)
    parser.add_argument("--vocab", type=str, default="rwkv_vocab_v20260603.txt")
    parser.add_argument("--chat-template", type=str, default="data/SFT/sample/chat_template.jinja")
    parser.add_argument("--n-epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ctx-len", "--ctx_len", dest="ctx_len", type=int, default=None)
    parser.add_argument("--pack", action="store_true", default=False)
    parser.add_argument("--pad", action="store_true", default=False)
    parser.add_argument("--pack-strategy", choices=["ordered", "best-fit-decreasing"], default="ordered")
    parser.add_argument("--pack-shard-group-size", type=int, default=1)
    parser.add_argument("--pack-length", type=int, default=None)
    parser.add_argument("--pad-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--current-date", type=str, default=None)
    parser.add_argument("--current-location", type=str, default=None)
    parser.add_argument("--add-generation-prompt", action="store_true", default=False)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.pack and args.pad:
        parser.error("--pack and --pad are mutually exclusive.")
    if args.pack_length is not None and args.pad_length is not None:
        parser.error("--pack-length and --pad-length are mutually exclusive.")
    if (args.pack or args.pad) and args.ctx_len is None:
        parser.error("--ctx-len is required when using --pack or --pad.")
    if args.pack and args.pack_length is not None:
        parser.error("--pack cannot be used with --pack-length.")
    if args.pad and args.pad_length is not None:
        parser.error("--pad cannot be used with --pad-length.")
    if args.ctx_len is not None and args.ctx_len <= 0:
        parser.error("--ctx-len must be a positive integer.")
    if args.pack_shard_group_size <= 0:
        parser.error("--pack-shard-group-size must be a positive integer.")
    if args.pack_length is not None and args.pack_length <= 0:
        parser.error("--pack-length must be a positive integer.")
    if args.pad_length is not None and args.pad_length <= 0:
        parser.error("--pad-length must be a positive integer.")

    pack_length = args.pack_length
    pad_length = args.pad_length
    if args.pack:
        if pad_length is not None:
            parser.error("--pack cannot be used with --pad-length.")
        pack_length = args.ctx_len + 1
    if args.pad:
        if pack_length is not None:
            parser.error("--pad cannot be used with --pack-length.")
        pad_length = args.ctx_len + 1

    stats = build_binidx_dataset(
        args.input_jsonl,
        output_prefix=args.out_prefix,
        vocab_path=args.vocab,
        template_path=args.chat_template,
        n_epoch=args.n_epoch,
        seed=args.seed,
        pack_length=pack_length,
        pad_length=pad_length,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pack_strategy=args.pack_strategy,
        pack_shard_group_size=args.pack_shard_group_size,
        current_date=args.current_date,
        current_location=args.current_location,
    )

    print(
        "### Built SFT binidx dataset: "
        f"prefix={stats['output_prefix']} "
        f"docs={stats['documents']} "
        f"tokens={stats['tokens']} "
        f"trainable_tokens={stats['trainable_tokens']}"
    )


if __name__ == "__main__":
    main()

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
    parser.add_argument("input_jsonl", type=str)
    parser.add_argument("--out-prefix", type=str, default=None)
    parser.add_argument("--vocab", type=str, default="rwkv_vocab_v20260603.txt")
    parser.add_argument("--chat-template", type=str, default="data/SFT/sample/chat_template.jinja")
    parser.add_argument("--n-epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pack-length", type=int, default=None)
    parser.add_argument("--current-date", type=str, default=None)
    parser.add_argument("--current-location", type=str, default=None)
    parser.add_argument("--add-generation-prompt", action="store_true", default=False)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stats = build_binidx_dataset(
        args.input_jsonl,
        output_prefix=args.out_prefix,
        vocab_path=args.vocab,
        template_path=args.chat_template,
        n_epoch=args.n_epoch,
        seed=args.seed,
        pack_length=args.pack_length,
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

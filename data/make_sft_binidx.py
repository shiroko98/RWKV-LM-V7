from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sft_binidx import build_binidx_dataset  # noqa: E402


class ProgressBar:
    def __init__(self, *, enabled: bool = True, interval: float = 0.2, stream=None):
        self.enabled = enabled
        self.interval = interval
        self.stream = stream or sys.stderr
        self.start_time = time.perf_counter()
        self.last_update = 0.0
        self.last_line_len = 0
        self.rendered = False
        self.finished = False

    def __call__(self, event: dict[str, object]):
        if not self.enabled or self.finished:
            return
        done = int(event.get("done", 0) or 0)
        total = event.get("total")
        total_int = int(total) if total is not None else 0
        now = time.perf_counter()
        is_first = done <= 1
        is_final = total_int > 0 and done >= total_int
        if not is_first and not is_final and now - self.last_update < self.interval:
            return

        elapsed = max(now - self.start_time, 1e-9)
        speed = done / elapsed
        remaining = max(total_int - done, 0) if total_int > 0 else 0
        stage = str(event.get("stage", "progress"))
        unit = str(event.get("unit", "samples"))
        path = str(event.get("source_path", ""))
        line_number = event.get("line_number")
        source = f"{path}:{line_number}" if line_number else path
        source = self._shorten_source(source)

        if total_int > 0:
            ratio = min(max(done / total_int, 0.0), 1.0)
            width = 24
            filled = int(ratio * width)
            bar = "#" * filled + "-" * (width - filled)
            count_text = f"{done}/{total_int}"
        else:
            bar = "-" * 24
            count_text = str(done)

        group_index = event.get("group_index")
        group_count = event.get("group_count")
        group_text = ""
        if group_index is not None and group_count is not None:
            group_text = f" group={group_index}/{group_count}"

        line = (
            f"### SFT progress {stage} [{bar}] {count_text} "
            f"left={remaining} speed={speed:.1f} {unit}/s{group_text} file={source}"
        )
        self.stream.write("\r" + line + self._clear_suffix(line))
        self.stream.flush()
        self.last_update = now
        self.last_line_len = len(line)
        self.rendered = True

    def finish(self):
        if not self.enabled or self.finished:
            return
        if self.rendered:
            self.stream.write("\n")
            self.stream.flush()
        self.finished = True

    def _clear_suffix(self, line: str) -> str:
        suffix_len = max(self.last_line_len - len(line), 0)
        return " " * suffix_len

    @staticmethod
    def _shorten_source(source: str, limit: int = 80) -> str:
        if len(source) <= limit:
            return source
        return "..." + source[-(limit - 3):]


class ErrorLog:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.file = None

    def __enter__(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.file is not None:
            self.file.close()
        return False

    def __call__(self, event: dict[str, object]):
        if self.file is None:
            return
        self.file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self.file.flush()


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
    parser.add_argument("--pack-cache-dir", type=str, default=None)
    parser.add_argument("--pack-length", type=int, default=None)
    parser.add_argument("--pad-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=64)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=float, default=0.2)
    parser.add_argument("--error-log", type=str, default=None)
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
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative.")
    if args.worker_chunksize <= 0:
        parser.error("--worker-chunksize must be a positive integer.")
    if args.pack_cache_dir is not None:
        if not args.pack or args.pack_strategy != "best-fit-decreasing":
            parser.error("--pack-cache-dir requires --pack --pack-strategy best-fit-decreasing.")
        if args.shuffle:
            parser.error("--pack-cache-dir requires --no-shuffle for deterministic cache reuse.")

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

    progress_bar = ProgressBar(enabled=args.progress, interval=args.progress_interval)
    with ErrorLog(args.error_log) as error_log:
        try:
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
                worker_chunksize=args.worker_chunksize,
                pack_cache_dir=args.pack_cache_dir,
                current_date=args.current_date,
                current_location=args.current_location,
                progress_callback=progress_bar,
                error_callback=error_log,
            )
        finally:
            progress_bar.finish()

    print(
        "### Built SFT binidx dataset: "
        f"prefix={stats['output_prefix']} "
        f"docs={stats['documents']} "
        f"tokens={stats['tokens']} "
        f"trainable_tokens={stats['trainable_tokens']}"
    )


if __name__ == "__main__":
    main()

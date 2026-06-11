#!/usr/bin/env python3
"""Convert a DeepSpeed checkpoint and run a prompt smoke test.

This is a thin wrapper around:
  - scripts/convert_deepspeed_checkpoint_to_pth.py
  - scripts/run_converted_rwkv_demo.py

It keeps conversion and prompt inference in one command while still reusing the
existing implementation for both steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERT_SCRIPT = REPO_ROOT / "scripts" / "convert_deepspeed_checkpoint_to_pth.py"
DEMO_SCRIPT = REPO_ROOT / "scripts" / "run_converted_rwkv_demo.py"


def default_output_path(checkpoint_dir: Path, dtype_name: str) -> Path:
    return checkpoint_dir.parent / f"{checkpoint_dir.stem}.{dtype_name}.pth"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="DeepSpeed/ZeRO checkpoint directory, for example rwkv-step-1000.pth")
    parser.add_argument("--output-file", default="", help="Converted single-file .pth. Defaults to a sibling file like rwkv-step-1000.bf16.pth")
    parser.add_argument("--convert-dtype", choices=("bf16", "fp16", "fp32"), default="bf16", help="Floating dtype for the converted .pth")
    parser.add_argument("--summary-file", default="", help="Optional parameter summary file written during conversion")
    parser.add_argument("--verify-summary-file", default="", help="Optional reference summary file for conversion verification")
    parser.add_argument("--tag", default=None, help="Explicit DeepSpeed tag. Leave empty to use checkpoint latest")
    parser.add_argument("--exclude-frozen-parameters", action="store_true", help="Forward to the DeepSpeed converter when supported")
    parser.add_argument("--lazy-mode", action=argparse.BooleanOptionalAction, default=True, help="Forward lazy-mode choice to the DeepSpeed converter")
    parser.add_argument("--force-convert", action="store_true", help="Convert even when --output-file already exists")
    parser.add_argument("--skip-demo", action="store_true", help="Only convert/reuse the .pth and skip prompt inference")

    parser.add_argument("--vocab-path", default=str(REPO_ROOT / "rwkv_vocab_v20260603.txt"), help="Tokenizer vocab for the demo")
    parser.add_argument("--prompt", default="你好，请用一句话介绍 RWKV。", help="User prompt. Rendered through --chat-template unless --raw-prompt is set")
    parser.add_argument("--chat-template", default=str(REPO_ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"), help="SFT chat template for prompt rendering")
    parser.add_argument("--raw-prompt", action="store_true", help="Use --prompt directly without chat-template rendering")
    parser.add_argument("--system-prompt", default="", help="Optional system message for chat-template rendering")
    parser.add_argument("--current-date", default="", help="Optional current_date field for chat-template rendering")
    parser.add_argument("--current-location", default="", help="Optional current_location field for chat-template rendering")
    parser.add_argument("--add-generation-prompt", action=argparse.BooleanOptionalAction, default=True, help="Append assistant generation prompt when rendering chat messages")
    parser.add_argument("--enable-thinking", action="store_true", help="Open a <think> block in the assistant generation prompt")
    parser.add_argument("--no-add-thinking", action="store_true", help="Do not add an empty <think> block in the assistant generation prompt")
    parser.add_argument("--device", default="cuda", help="Inference device for the demo")
    parser.add_argument("--runtime-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto", help="Runtime dtype for inference. 'auto' follows checkpoint dtype")
    parser.add_argument("--topk", type=int, default=10, help="Top next-token candidates to print")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when --sample is set")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p cutoff when --sample is set. 0 disables it")
    parser.add_argument("--sample", action="store_true", help="Sample generated tokens instead of greedy decoding")
    return parser.parse_args(argv)


def resolve_output_file(args: argparse.Namespace) -> Path:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if args.output_file:
        return Path(args.output_file).expanduser().resolve()
    return default_output_path(checkpoint_dir, args.convert_dtype).resolve()


def build_convert_command(args: argparse.Namespace, output_file: Path) -> list[str]:
    command = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--checkpoint-dir",
        str(Path(args.checkpoint_dir).expanduser().resolve()),
        "--output-file",
        str(output_file),
        "--dtype",
        args.convert_dtype,
    ]
    if args.summary_file:
        command.extend(["--summary-file", str(Path(args.summary_file).expanduser().resolve())])
    if args.verify_summary_file:
        command.extend(["--verify-summary-file", str(Path(args.verify_summary_file).expanduser().resolve())])
    if args.tag is not None:
        command.extend(["--tag", args.tag])
    if args.exclude_frozen_parameters:
        command.append("--exclude-frozen-parameters")
    command.append("--lazy-mode" if args.lazy_mode else "--no-lazy-mode")
    return command


def build_demo_command(args: argparse.Namespace, model_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(DEMO_SCRIPT),
        "--model-path",
        str(model_path),
        "--vocab-path",
        str(Path(args.vocab_path).expanduser().resolve()),
        "--prompt",
        args.prompt,
        "--chat-template",
        str(Path(args.chat_template).expanduser().resolve()),
        "--device",
        args.device,
        "--dtype",
        args.runtime_dtype,
        "--topk",
        str(args.topk),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    if args.raw_prompt:
        command.append("--raw-prompt")
    if args.system_prompt:
        command.extend(["--system-prompt", args.system_prompt])
    if args.current_date:
        command.extend(["--current-date", args.current_date])
    if args.current_location:
        command.extend(["--current-location", args.current_location])
    command.append("--add-generation-prompt" if args.add_generation_prompt else "--no-add-generation-prompt")
    if args.enable_thinking:
        command.append("--enable-thinking")
    if args.no_add_thinking:
        command.append("--no-add-thinking")
    if args.sample:
        command.append("--sample")
    return command


def run_command(command: list[str], label: str) -> int:
    print(f"[convert-demo] {label}:")
    print(" ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT)
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_file = resolve_output_file(args)

    if not checkpoint_dir.is_dir():
        print(f"[convert-demo] checkpoint dir not found: {checkpoint_dir}", file=sys.stderr)
        return 2

    if output_file.exists() and not args.force_convert:
        print(f"[convert-demo] reusing existing converted checkpoint: {output_file}")
    else:
        rc = run_command(build_convert_command(args, output_file), "convert DeepSpeed checkpoint")
        if rc != 0:
            return rc

    if args.skip_demo:
        print(f"[convert-demo] skipped demo; converted checkpoint: {output_file}")
        return 0

    return run_command(build_demo_command(args, output_file), "run prompt demo")


if __name__ == "__main__":
    raise SystemExit(main())

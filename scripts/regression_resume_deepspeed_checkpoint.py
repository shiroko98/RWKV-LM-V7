#!/usr/bin/env python3
"""Smoke-test resuming training from a real DeepSpeed sharded checkpoint.

Example:
  python scripts/regression_resume_deepspeed_checkpoint.py \
    --checkpoint-path /path/to/outs/rwkv-40.pth \
    --log-file /tmp/rwkv-resume-smoke.log \
    -- python train.py --load_model /path/to/outs/rwkv-40.pth ...

The script will:
  1. validate that the checkpoint path looks like a DeepSpeed sharded checkpoint directory
  2. launch the provided training command
  3. watch stdout/stderr for resume markers
  4. stop the process group cleanly once resume looks healthy
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def looks_like_deepspeed_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir() or path.suffix != ".pth":
        return False
    try:
        entries = {entry.name for entry in path.iterdir()}
    except OSError:
        return False
    if "latest" in entries:
        return True
    return any(name.endswith("_model_states.pt") or name.endswith("_optim_states.pt") for name in entries)


def default_require_patterns() -> list[str]:
    return [
        "Resuming trainer state from",
        "Restoring states from the checkpoint path at",
    ]


def default_progress_patterns() -> list[str]:
    return ["loss=", "REAL it/s", "Kt/s", "Epoch "]


def line_has_failure_marker(line: str) -> bool:
    failure_markers = (
        "Traceback (most recent call last):",
        "ChildFailedError",
        "KeyError:",
        "RuntimeError:",
        "AssertionError:",
        "[resume-smoke] FAIL",
    )
    return any(marker in line for marker in failure_markers)


def start_process(command: list[str]) -> subprocess.Popen[str]:
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def _send_group_signal(process: subprocess.Popen[str], sig: int) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            if sig == signal.SIGINT and hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        else:
            os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop_process_group(process: subprocess.Popen[str], interrupt_timeout: int, term_timeout: int, kill_timeout: int) -> None:
    if process.poll() is not None:
        return

    _send_group_signal(process, signal.SIGINT)
    try:
        process.wait(timeout=interrupt_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    _send_group_signal(process, signal.SIGTERM)
    try:
        process.wait(timeout=term_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    _send_group_signal(process, signal.SIGKILL)
    try:
        process.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        pass


def reader_thread(stdout, log_handle, line_queue: queue.Queue[str]) -> None:
    try:
        for raw_line in iter(stdout.readline, ""):
            line = raw_line.rstrip("\n")
            log_handle.write(raw_line)
            log_handle.flush()
            line_queue.put(line)
    finally:
        stdout.close()


def tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-limit:]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, help="Path to the DeepSpeed sharded checkpoint directory, usually something like rwkv-40.pth")
    parser.add_argument("--log-file", default="", help="Where to write the captured launcher output. Defaults to ./resume-smoke-<timestamp>.log")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Fail if resume has not looked healthy within this many seconds")
    parser.add_argument("--steady-seconds", type=int, default=120, help="How long to keep waiting after restore markers appear before timing out. This no longer counts as a pass by itself.")
    parser.add_argument("--interrupt-timeout", type=int, default=30)
    parser.add_argument("--term-timeout", type=int, default=15)
    parser.add_argument("--kill-timeout", type=int, default=5)
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument(
        "--require-pattern",
        action="append",
        default=[],
        help="Pattern that must appear in the log. Can be passed multiple times. Defaults include both the train.py resume log and Lightning's checkpoint-restore log.",
    )
    parser.add_argument(
        "--progress-pattern",
        action="append",
        default=[],
        help="Pattern that suggests training has really entered the loop. Can be passed multiple times. Defaults to ['loss=', 'REAL it/s', 'Kt/s', 'Epoch ']",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Training command to run, prefixed by --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing training command after --")
    return args


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not looks_like_deepspeed_checkpoint_dir(checkpoint_path):
        print(f"[resume-smoke] checkpoint does not look like a DeepSpeed sharded checkpoint directory: {checkpoint_path}", file=sys.stderr)
        return 2

    require_patterns = args.require_pattern or default_require_patterns()
    progress_patterns = args.progress_pattern or default_progress_patterns()

    log_path = Path(args.log_file).expanduser() if args.log_file else Path.cwd() / f"resume-smoke-{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[resume-smoke] checkpoint: {checkpoint_path}")
    print(f"[resume-smoke] log file:   {log_path}")
    print(f"[resume-smoke] command:    {' '.join(args.command)}")

    start_time = time.monotonic()
    pattern_hits = {pattern: False for pattern in require_patterns}
    saw_progress = False
    resume_seen_at = None
    failure_markers = []

    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = start_process(args.command)
        assert process.stdout is not None
        lines: queue.Queue[str] = queue.Queue()
        worker = threading.Thread(target=reader_thread, args=(process.stdout, log_handle, lines), daemon=True)
        worker.start()

        passed = False
        timeout_hit = False

        while True:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                line = None

            if line is not None:
                print(line)
                for pattern in require_patterns:
                    if pattern in line:
                        pattern_hits[pattern] = True
                if any(pattern in line for pattern in progress_patterns):
                    saw_progress = True
                if line_has_failure_marker(line):
                    failure_markers.append(line)
                if resume_seen_at is None and all(pattern_hits.values()):
                    resume_seen_at = time.monotonic()

            if process.poll() is not None:
                break

            if all(pattern_hits.values()) and saw_progress:
                passed = True
                break

            if time.monotonic() - start_time >= args.timeout_seconds:
                timeout_hit = True
                break

        if passed:
            print("[resume-smoke] success markers observed, stopping process group cleanly...")
        elif timeout_hit:
            print("[resume-smoke] timeout waiting for resume success markers", file=sys.stderr)
        elif process.poll() is not None:
            print(f"[resume-smoke] process exited early with code {process.returncode}", file=sys.stderr)

        stop_process_group(process, args.interrupt_timeout, args.term_timeout, args.kill_timeout)
        worker.join(timeout=2)
        remaining_output = process.stdout.read() if process.stdout and not process.stdout.closed else ""
        if remaining_output:
            log_handle.write(remaining_output)
            print(remaining_output, end="")

    if passed:
        print("[resume-smoke] PASS")
        return 0

    print("[resume-smoke] FAIL", file=sys.stderr)
    if failure_markers:
        print("[resume-smoke] failure markers detected:", file=sys.stderr)
        for marker in failure_markers:
            print(f"  {marker}", file=sys.stderr)

    tail = tail_lines(log_path, args.tail_lines)
    if tail:
        print("[resume-smoke] log tail:", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

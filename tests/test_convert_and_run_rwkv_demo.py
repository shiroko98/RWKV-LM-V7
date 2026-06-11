import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_and_run_rwkv_demo.py"
SPEC = importlib.util.spec_from_file_location("convert_and_run_rwkv_demo", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


def _checkpoint_dir(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "rwkv-step-1000.pth"
    checkpoint.mkdir()
    (checkpoint / "latest").write_text("global_step1000", encoding="utf-8")
    return checkpoint


def test_default_output_path_uses_checkpoint_stem(tmp_path: Path):
    checkpoint = tmp_path / "rwkv-step-1000.pth"

    assert script.default_output_path(checkpoint, "bf16") == tmp_path / "rwkv-step-1000.bf16.pth"


def test_build_commands_forward_conversion_and_demo_options(tmp_path: Path):
    checkpoint = _checkpoint_dir(tmp_path)
    output = tmp_path / "merged.pth"
    summary = tmp_path / "merged.summary.txt"
    vocab = tmp_path / "vocab.txt"
    template = tmp_path / "chat_template.jinja"

    args = script.parse_args(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--output-file",
            str(output),
            "--convert-dtype",
            "fp16",
            "--summary-file",
            str(summary),
            "--no-lazy-mode",
            "--vocab-path",
            str(vocab),
            "--chat-template",
            str(template),
            "--prompt",
            "你好",
            "--system-prompt",
            "系统",
            "--current-date",
            "2026-06-11",
            "--current-location",
            "Shanghai",
            "--device",
            "cpu",
            "--runtime-dtype",
            "fp32",
            "--topk",
            "3",
            "--max-new-tokens",
            "5",
            "--top-p",
            "0.5",
            "--sample",
        ]
    )

    convert_command = script.build_convert_command(args, output.resolve())
    demo_command = script.build_demo_command(args, output.resolve())

    assert convert_command[:2] == [sys.executable, str(script.CONVERT_SCRIPT)]
    assert "--checkpoint-dir" in convert_command
    assert str(checkpoint.resolve()) in convert_command
    assert convert_command[convert_command.index("--dtype") + 1] == "fp16"
    assert "--summary-file" in convert_command
    assert "--no-lazy-mode" in convert_command

    assert demo_command[:2] == [sys.executable, str(script.DEMO_SCRIPT)]
    assert demo_command[demo_command.index("--model-path") + 1] == str(output.resolve())
    assert demo_command[demo_command.index("--prompt") + 1] == "你好"
    assert demo_command[demo_command.index("--chat-template") + 1] == str(template.resolve())
    assert demo_command[demo_command.index("--vocab-path") + 1] == str(vocab.resolve())
    assert demo_command[demo_command.index("--dtype") + 1] == "fp32"
    assert "--sample" in demo_command
    assert "--add-generation-prompt" in demo_command


def test_build_commands_forward_optional_flags(tmp_path: Path):
    checkpoint = _checkpoint_dir(tmp_path)
    output = tmp_path / "merged.pth"
    reference = tmp_path / "reference.summary.txt"

    args = script.parse_args(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--output-file",
            str(output),
            "--verify-summary-file",
            str(reference),
            "--tag",
            "global_step1000",
            "--exclude-frozen-parameters",
            "--raw-prompt",
            "--no-add-generation-prompt",
            "--enable-thinking",
            "--no-add-thinking",
        ]
    )

    convert_command = script.build_convert_command(args, output.resolve())
    demo_command = script.build_demo_command(args, output.resolve())

    assert convert_command[convert_command.index("--verify-summary-file") + 1] == str(reference.resolve())
    assert convert_command[convert_command.index("--tag") + 1] == "global_step1000"
    assert "--exclude-frozen-parameters" in convert_command
    assert "--raw-prompt" in demo_command
    assert "--no-add-generation-prompt" in demo_command
    assert "--enable-thinking" in demo_command
    assert "--no-add-thinking" in demo_command


def test_run_command_invokes_subprocess(monkeypatch):
    calls = []

    def fake_run(command, cwd):
        calls.append((command, cwd))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    rc = script.run_command(["python", "x.py"], "demo")

    assert rc == 7
    assert calls == [(["python", "x.py"], script.REPO_ROOT)]


def test_main_stops_when_conversion_fails(tmp_path: Path, monkeypatch):
    checkpoint = _checkpoint_dir(tmp_path)
    output = tmp_path / "merged.bf16.pth"
    calls = []

    def fake_run(command, label):
        calls.append(label)
        return 9

    monkeypatch.setattr(script, "run_command", fake_run)

    rc = script.main(["--checkpoint-dir", str(checkpoint), "--output-file", str(output)])

    assert rc == 9
    assert calls == ["convert DeepSpeed checkpoint"]


def test_main_reuses_existing_pth_and_runs_demo(tmp_path: Path, monkeypatch):
    checkpoint = _checkpoint_dir(tmp_path)
    output = tmp_path / "merged.bf16.pth"
    output.write_bytes(b"already converted")
    calls = []

    monkeypatch.setattr(script, "run_command", lambda command, label: calls.append((label, command)) or 0)

    rc = script.main([
        "--checkpoint-dir",
        str(checkpoint),
        "--output-file",
        str(output),
        "--prompt",
        "hello",
    ])

    assert rc == 0
    assert [label for label, _ in calls] == ["run prompt demo"]


def test_main_force_convert_and_skip_demo(tmp_path: Path, monkeypatch):
    checkpoint = _checkpoint_dir(tmp_path)
    output = tmp_path / "merged.bf16.pth"
    output.write_bytes(b"old")
    calls = []

    monkeypatch.setattr(script, "run_command", lambda command, label: calls.append((label, command)) or 0)

    rc = script.main([
        "--checkpoint-dir",
        str(checkpoint),
        "--output-file",
        str(output),
        "--force-convert",
        "--skip-demo",
    ])

    assert rc == 0
    assert [label for label, _ in calls] == ["convert DeepSpeed checkpoint"]


def test_main_rejects_missing_checkpoint_dir(tmp_path: Path):
    rc = script.main(["--checkpoint-dir", str(tmp_path / "missing")])

    assert rc == 2

import importlib.util
import sys
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_converted_rwkv_demo.py"
SPEC = importlib.util.spec_from_file_location("run_converted_rwkv_demo_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
demo_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo_script)


def test_infer_model_dims_from_state_dict():
    state_dict = OrderedDict(
        [
            ("emb.weight", torch.zeros(65536, 4096, dtype=torch.bfloat16)),
            ("blocks.0.ln1.weight", torch.zeros(4096, dtype=torch.bfloat16)),
            ("blocks.1.ln1.weight", torch.zeros(4096, dtype=torch.bfloat16)),
            ("blocks.0.att.w1", torch.zeros(4096, 192, dtype=torch.bfloat16)),
            ("blocks.0.att.a1", torch.zeros(4096, 192, dtype=torch.bfloat16)),
            ("blocks.0.att.v1", torch.zeros(4096, 128, dtype=torch.bfloat16)),
            ("blocks.0.att.g1", torch.zeros(4096, 384, dtype=torch.bfloat16)),
            ("blocks.0.att.r_k", torch.zeros(64, 64, dtype=torch.bfloat16)),
        ]
    )

    dims = demo_script.infer_model_dims_from_state_dict(state_dict)

    assert dims == {
        "n_layer": 2,
        "n_embd": 4096,
        "vocab_size": 65536,
        "d_decay_lora": 192,
        "d_aaa_lora": 192,
        "d_mv_lora": 128,
        "d_gate_lora": 384,
        "head_size": 64,
        "n_head": 64,
    }


def test_infer_runtime_dtype_prefers_checkpoint_dtype_in_auto_mode():
    state_dict = OrderedDict([("emb.weight", torch.zeros(2, 2, dtype=torch.bfloat16))])

    assert demo_script.infer_runtime_dtype(state_dict, "auto") == torch.bfloat16
    assert demo_script.infer_runtime_dtype(state_dict, "fp32") == torch.float32


def test_resolve_prompt_renders_user_prompt_with_chat_template():
    args = Namespace(
        raw_prompt=False,
        prompt="你好",
        chat_template=str(REPO_ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"),
        system_prompt="系统提示",
        current_date="2026-06-09",
        current_location="Shanghai",
        add_generation_prompt=True,
        enable_thinking=False,
        no_add_thinking=False,
    )

    prompt = demo_script.resolve_prompt(args)

    assert "<|im_start|>System: 系统提示" in prompt
    assert "Current date: 2026-06-09" in prompt
    assert "Current location: Shanghai" in prompt
    assert "<|im_start|>User: 你好<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>Assistant: <think>\n\n</think>\n\n")


def test_parse_args_uses_plain_prompt_not_message_files(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_converted_rwkv_demo.py",
            "--model-path",
            "model.pth",
            "--prompt",
            "你好",
        ],
    )

    args = demo_script.parse_args()

    assert args.prompt == "你好"
    assert args.raw_prompt is False
    assert not hasattr(args, "messages_file")
    assert not hasattr(args, "messages_json")


def test_resolve_prompt_can_use_raw_prompt_and_rejects_thinking_conflict():
    args = Namespace(raw_prompt=True, prompt="raw text", enable_thinking=True, no_add_thinking=True)
    assert demo_script.resolve_prompt(args) == "raw text"

    args.raw_prompt = False
    args.chat_template = str(REPO_ROOT / "data" / "SFT" / "sample" / "chat_template.jinja")
    args.system_prompt = ""
    args.current_date = ""
    args.current_location = ""
    args.add_generation_prompt = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        demo_script.resolve_prompt(args)


def test_runtime_block_only_creates_ln0_for_first_layer():
    args = type("Args", (), {"n_embd": 128, "dim_att": 128, "dim_ffn": 512, "head_size_a": 64, "n_layer": 2})()

    block0 = demo_script.runtime.Block(args, 0)
    block1 = demo_script.runtime.Block(args, 1)

    assert hasattr(block0, "ln0")
    assert not hasattr(block1, "ln0")

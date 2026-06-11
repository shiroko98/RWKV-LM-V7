from argparse import Namespace
from pathlib import Path

import pytest

from src.sft_prompt import render_user_prompt, resolve_prompt_from_args


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"


def test_render_user_prompt_builds_one_turn_chat_prompt():
    prompt = render_user_prompt(
        prompt="你好",
        chat_template=str(TEMPLATE_PATH),
        system_prompt="系统提示",
        current_date="2026-06-09",
        current_location="Shanghai",
    )

    assert "<|im_start|>System: 系统提示" in prompt
    assert "Current date: 2026-06-09" in prompt
    assert "Current location: Shanghai" in prompt
    assert "<|im_start|>User: 你好<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>Assistant: <think>\n\n</think>\n\n")


def test_render_user_prompt_supports_raw_prompt():
    assert render_user_prompt(prompt="raw text", chat_template="", raw_prompt=True) == "raw text"


def test_render_user_prompt_rejects_thinking_conflict():
    with pytest.raises(ValueError, match="mutually exclusive"):
        render_user_prompt(
            prompt="你好",
            chat_template=str(TEMPLATE_PATH),
            enable_thinking=True,
            no_add_thinking=True,
        )


def test_render_user_prompt_force_thinking_opens_think_block():
    prompt = render_user_prompt(
        prompt="你好",
        chat_template=str(TEMPLATE_PATH),
        force_thinking=True,
    )

    assert prompt.endswith("<|im_start|>Assistant: <think>\n")
    assert "</think>" not in prompt.rsplit("<|im_start|>Assistant:", 1)[1]


def test_render_user_prompt_requires_template_file():
    with pytest.raises(FileNotFoundError, match="chat template file not found"):
        render_user_prompt(prompt="你好", chat_template=str(REPO_ROOT / "missing.jinja"))


def test_resolve_prompt_from_args_uses_defaults_for_optional_fields():
    args = Namespace(
        prompt="你好",
        chat_template=str(TEMPLATE_PATH),
    )

    prompt = resolve_prompt_from_args(args)

    assert "<|im_start|>User: 你好<|im_end|>" in prompt


def test_resolve_prompt_from_args_keeps_raw_prompt_without_template():
    args = Namespace(prompt="raw text", raw_prompt=True)

    assert resolve_prompt_from_args(args) == "raw text"

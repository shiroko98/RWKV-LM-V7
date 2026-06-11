from __future__ import annotations

from pathlib import Path
from typing import Any

from src.sft_binidx import load_chat_template, render_chat_template


def render_user_prompt(
    *,
    prompt: str,
    chat_template: str,
    raw_prompt: bool = False,
    system_prompt: str = "",
    current_date: str = "",
    current_location: str = "",
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    force_thinking: bool = False,
    no_add_thinking: bool = False,
) -> str:
    if raw_prompt:
        return prompt
    enable_thinking = enable_thinking or force_thinking
    if enable_thinking and no_add_thinking:
        raise ValueError("--enable-thinking/--force-thinking and --no-add-thinking are mutually exclusive.")

    template_path = Path(chat_template).expanduser().resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"chat template file not found: {template_path}")

    messages: list[dict[str, Any]] = []
    if system_prompt or current_date or current_location:
        system_message: dict[str, Any] = {"role": "system"}
        if system_prompt:
            system_message["content"] = system_prompt
        if current_date:
            system_message["current_date"] = current_date
        if current_location:
            system_message["current_location"] = current_location
        messages.append(system_message)
    messages.append({"role": "user", "content": prompt})

    return render_chat_template(
        load_chat_template(str(template_path)),
        messages,
        tools=None,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        no_add_thinking=no_add_thinking,
    )


def resolve_prompt_from_args(args: Any) -> str:
    return render_user_prompt(
        prompt=args.prompt,
        chat_template=getattr(args, "chat_template", ""),
        raw_prompt=getattr(args, "raw_prompt", False),
        system_prompt=getattr(args, "system_prompt", ""),
        current_date=getattr(args, "current_date", ""),
        current_location=getattr(args, "current_location", ""),
        add_generation_prompt=getattr(args, "add_generation_prompt", True),
        enable_thinking=getattr(args, "enable_thinking", False),
        force_thinking=getattr(args, "force_thinking", False),
        no_add_thinking=getattr(args, "no_add_thinking", False),
    )

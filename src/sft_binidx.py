from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from bisect import bisect_left, insort
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from jinja2 import Environment

from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.binidx import MMapIndexedDataset


IM_START_TOKEN = "<|im_start|>"
ASSISTANT_PREFIX = "<|im_start|>Assistant: "
IM_END_TOKEN = "<|im_end|>"
EOD_TOKEN = "<|endoftext|>"
NO_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def index_file_path(prefix_path: str) -> str:
    return prefix_path + ".idx"


def data_file_path(prefix_path: str) -> str:
    return prefix_path + ".bin"


def mask_prefix_path(prefix_path: str) -> str:
    return prefix_path + ".mask"


def mask_file_path(prefix_path: str) -> str:
    return data_file_path(mask_prefix_path(prefix_path))


class MMapIndexedDatasetBuilder:
    def __init__(self, out_file: str, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes: list[int] = []
        self._doc_idx = [0]

    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self, index_file: str):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


@dataclass(frozen=True)
class Segment:
    text: str
    trainable: bool


@dataclass(frozen=True)
class EncodedDocument:
    input_ids: list[int]
    loss_mask: list[int]


@dataclass(frozen=True)
class JsonlSourceLine:
    text: str
    source_path: str
    line_number: int


@dataclass
class FilterStats:
    filtered: int = 0


class SFTDocumentBuildError(ValueError):
    def __init__(self, event: dict[str, object]):
        self.event = event
        location = f"{event.get('source_path', '<unknown>')}:{event.get('line_number', '?')}"
        error_type = event.get("error_type", "Error")
        error_message = event.get("error_message", "")
        super().__init__(f"Failed to build SFT document from {location}: {error_type}: {error_message}")

    def __reduce__(self):
        return (type(self), (self.event,))


def _tojson_filter(obj, ensure_ascii=False):
    return json.dumps(obj, ensure_ascii=ensure_ascii)


def _build_jinja_env() -> Environment:
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson_filter
    return env


def eod_token_id(tokenizer: TRIE_TOKENIZER) -> int:
    token_ids = tokenizer.encode(EOD_TOKEN)
    if len(token_ids) != 1:
        raise ValueError(f"{EOD_TOKEN} must encode to exactly one token for packing.")
    return token_ids[0]


def default_system_message() -> str:
    return "You are a helpful assistant. Your name is xiaoke and is built by CETC."


def visible_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                pieces.append(str(item["text"]))
            elif item is not None:
                pieces.append(str(item))
        return "".join(pieces)
    if content is None:
        return ""
    return str(content)


def build_system_text(
    system_message: dict | None,
    *,
    current_date: str | None = None,
    current_location: str | None = None,
) -> str:
    if system_message and system_message.get("content"):
        text = visible_text(system_message.get("content"))
    else:
        text = default_system_message()

    resolved_date = current_date
    resolved_location = current_location
    if system_message:
        if resolved_date is None:
            resolved_date = system_message.get("current_date")
        if resolved_location is None:
            resolved_location = system_message.get("current_location")

    if resolved_date:
        text += f"\nCurrent date: {resolved_date}"
    if resolved_location:
        text += f"\nCurrent location: {resolved_location}"
    return text


def parse_tool_arguments(arguments):
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        if stripped.startswith("<parameter "):
            return stripped
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    if arguments is None:
        return {}
    return arguments


def normalize_tool_calls(tool_calls: Sequence[dict] | None):
    if not tool_calls:
        return tool_calls
    normalized = []
    for tool_call in tool_calls:
        entry = copy.deepcopy(tool_call)
        function = entry.get("function")
        if function and "arguments" in function:
            function["arguments"] = parse_tool_arguments(function.get("arguments"))
        normalized.append(entry)
    return normalized


def normalize_record(record: dict) -> dict:
    normalized = copy.deepcopy(record)
    for message in normalized.get("messages", []):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            message["tool_calls"] = normalize_tool_calls(message.get("tool_calls"))
    return normalized


def trim_trailing_tool_messages(messages: Sequence[dict]) -> list[dict]:
    trimmed = copy.deepcopy(list(messages))
    while trimmed and trimmed[-1].get("role") == "tool":
        trimmed.pop()
    return trimmed


def render_tool_schema(tools: Sequence[dict]) -> str:
    lines = ["<tools>"]
    for tool in tools:
        payload = tool.get("function", tool)
        lines.append(f"<tool>{json.dumps(payload, ensure_ascii=False, separators=(', ', ': '))}</tool>")
    lines.append("</tools>")
    return "\n".join(lines)


def render_tool_calls(tool_calls: Sequence[dict]) -> str:
    parts = ["<tool_call>"]
    for tool_call in tool_calls:
        payload = tool_call.get("function", tool_call)
        name = payload["name"]
        args = parse_tool_arguments(payload.get("arguments"))
        if isinstance(args, str):
            invoke = f"<invoke name=\"{name}\">{args}</invoke>"
        else:
            inner = []
            for key, value in args.items():
                encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                inner.append(f"<parameter name=\"{key}\">{encoded}</parameter>")
            invoke = f"<invoke name=\"{name}\">{''.join(inner)}</invoke>"
        parts.append(invoke)
    parts.append("</tool_call>")
    return "\n".join(parts)


def render_tool_response(message: dict) -> str:
    name_attr = f" name=\"{message['name']}\"" if message.get("name") else ""
    content = message.get("content")
    if isinstance(content, str):
        return f"<response{name_attr}>{content}</response>"

    responses = []
    for item in content or []:
        if isinstance(item, dict):
            body = item.get("output")
            if body is None and item.get("type") == "text":
                body = item.get("text", "")
            if body is None:
                body = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            body = str(item)
        responses.append(f"<response{name_attr}>{body}\n</response>")
    return "\n".join(responses)


def split_system_and_conversation(messages: Sequence[dict]):
    if messages and messages[0].get("role") == "system":
        return messages[0], list(messages[1:])
    return None, list(messages)


def last_assistant_content_index(messages: Sequence[dict]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    return None


def last_assistant_message(messages: Sequence[dict]) -> dict | None:
    index = last_assistant_content_index(messages)
    if index is None:
        return None
    return messages[index]


def _assistant_has_existing_think(content: str) -> bool:
    return "</think>" in content


def _preview_text(value, *, limit: int = 240) -> str:
    text = visible_text(value).replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _summarize_message_for_error(message: dict, index: int) -> dict[str, object]:
    if not isinstance(message, dict):
        return {
            "index": index,
            "role": type(message).__name__,
            "keys": [],
            "content_type": type(message).__name__,
            "content_length": len(str(message)),
            "content_preview": _preview_text(message),
            "has_think_close": False,
            "has_reasoning_content": False,
            "reasoning_content_length": 0,
            "tool_calls_count": 0,
        }
    content = visible_text(message.get("content"))
    reasoning_content = message.get("reasoning_content")
    reasoning_text = reasoning_content if isinstance(reasoning_content, str) else ""
    summary: dict[str, object] = {
        "index": index,
        "role": message.get("role"),
        "keys": sorted(str(key) for key in message.keys()),
        "content_type": type(message.get("content")).__name__,
        "content_length": len(content),
        "content_preview": _preview_text(content),
        "has_think_close": "</think>" in content,
        "has_reasoning_content": isinstance(reasoning_content, str),
        "reasoning_content_length": len(reasoning_text),
        "tool_calls_count": len(message.get("tool_calls") or []),
    }
    if reasoning_text:
        summary["reasoning_content_preview"] = _preview_text(reasoning_text)
    if message.get("name"):
        summary["name"] = message.get("name")
    return summary


def _summarize_messages_for_error(messages: Sequence[dict], *, edge_count: int = 8) -> list[dict[str, object]]:
    if len(messages) <= edge_count * 2:
        selected = list(enumerate(messages))
    else:
        selected = list(enumerate(messages[:edge_count]))
        selected.append((-1, {"role": "<omitted>", "content": f"{len(messages) - edge_count * 2} messages omitted"}))
        selected.extend((index, message) for index, message in enumerate(messages[-edge_count:], start=len(messages) - edge_count))
    return [_summarize_message_for_error(message, index) for index, message in selected]


def _summarize_record_for_error(record: dict | None) -> dict[str, object]:
    if record is None:
        return {}
    messages = record.get("messages")
    message_list = messages if isinstance(messages, list) else []
    assistant_indexes = [
        index for index, message in enumerate(message_list)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    user_indexes = [
        index for index, message in enumerate(message_list)
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return {
        "top_level_keys": sorted(str(key) for key in record.keys()),
        "message_count": len(message_list),
        "roles": [
            message.get("role") if isinstance(message, dict) else type(message).__name__
            for message in message_list
        ],
        "last_user_index": user_indexes[-1] if user_indexes else None,
        "last_assistant_index": assistant_indexes[-1] if assistant_indexes else None,
        "tools_count": len(record.get("tools") or []),
        "messages": _summarize_messages_for_error(message_list),
    }


def _source_error_event(
    source: JsonlSourceLine,
    exc: BaseException,
    *,
    record: dict | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "stage": "render-tokenize",
        "source_path": source.source_path,
        "line_number": source.line_number,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "source_text": source.text,
    }
    if record is None:
        event["raw_line_preview"] = _preview_text(source.text, limit=1000)
    else:
        event["record"] = record
        event["record_summary"] = _summarize_record_for_error(record)
    return event


def _emit_exception_error(
    error_callback,
    exc: BaseException,
    *,
    source: JsonlSourceLine | None = None,
) -> None:
    if error_callback is None:
        return
    if isinstance(exc, SFTDocumentBuildError):
        error_callback(exc.event)
        return
    event: dict[str, object] = {
        "stage": "render-tokenize",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    if source is not None:
        event["source_path"] = source.source_path
        event["line_number"] = source.line_number
        event["source_text"] = source.text
        event["raw_line_preview"] = _preview_text(source.text, limit=1000)
    error_callback(event)


def _messages_before_last_assistant(messages: Sequence[dict]) -> list[dict]:
    last_index = last_assistant_content_index(messages)
    if last_index is None:
        return copy.deepcopy(list(messages))
    return copy.deepcopy(list(messages[:last_index]))


def _messages_with_normalized_final_assistant(messages: Sequence[dict]) -> list[dict]:
    normalized_messages = copy.deepcopy(list(messages))
    last_index = last_assistant_content_index(normalized_messages)
    if last_index is None:
        return normalized_messages

    content = visible_text(normalized_messages[last_index].get("content"))
    if not _assistant_has_existing_think(content):
        normalized_messages[last_index]["content"] = NO_THINKING_PREFIX + content
    return normalized_messages


def _render_template(
    template,
    *,
    messages: Sequence[dict],
    tools: Sequence[dict] | None,
    add_generation_prompt: bool,
    enable_thinking: bool,
    no_add_thinking: bool,
) -> str:
    return template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        no_add_thinking=no_add_thinking,
        tools=list(tools or []),
    )


def _compute_trainable_span(rendered_text: str, prefix_text: str) -> tuple[int, int]:
    if not rendered_text.startswith(prefix_text):
        raise ValueError("Full rendered text must start with the rendered prefix text.")
    train_start = len(prefix_text)
    train_end = len(rendered_text)
    return train_start, train_end


def _char_mask_from_span(text: str, train_start: int, train_end: int) -> list[int]:
    mask = [0] * len(text)
    for index in range(train_start, train_end):
        mask[index] = 1
    return mask


def _tokenize_with_char_spans(tokenizer: TRIE_TOKENIZER, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = text.encode("utf-8")
    idx = 0
    tokens: list[int] = []
    byte_spans: list[tuple[int, int]] = []
    while idx < len(encoded):
        prev_idx = idx
        idx, _, values = tokenizer.root.find_longest(encoded, idx)
        if idx == prev_idx:
            raise AssertionError("Tokenizer failed to advance while encoding text.")
        _, token_id = next(iter(values))
        tokens.append(token_id)
        byte_spans.append((prev_idx, idx))

    byte_to_char = [0] * len(encoded)
    byte_index = 0
    for char_index, char in enumerate(text):
        char_bytes = char.encode("utf-8")
        for _ in range(len(char_bytes)):
            byte_to_char[byte_index] = char_index
            byte_index += 1

    char_spans = [(byte_to_char[start], byte_to_char[end - 1] + 1) for start, end in byte_spans]
    return tokens, char_spans


def _loss_mask_from_char_mask(char_mask: Sequence[int], char_spans: Sequence[tuple[int, int]]) -> list[int]:
    token_mask: list[int] = []
    for start, end in char_spans:
        if start == end:
            token_mask.append(0)
            continue
        token_mask.append(1 if any(char_mask[pos] for pos in range(start, end)) else 0)
    return token_mask


def build_template_segments(
    messages: Sequence[dict],
    *,
    tools: Sequence[dict] | None = None,
    current_date: str | None = None,
    current_location: str | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    no_add_thinking: bool = False,
) -> list[Segment]:
    system_message, conversation_messages = split_system_and_conversation(messages)
    last_assistant_idx = last_assistant_content_index(messages)

    segments: list[Segment] = []
    system_message = copy.deepcopy(system_message) if system_message else None
    if system_message is not None:
        if current_date is not None:
            system_message["current_date"] = current_date
        if current_location is not None:
            system_message["current_location"] = current_location

    segments.append(Segment(f"{IM_START_TOKEN}System: ", False))
    segments.append(Segment(build_system_text(system_message), False))
    if tools:
        segments.append(
            Segment(
                "\n\n# Tools\n"
                "You may call one or more tools to assist with the user query.\n"
                "Here are the tools available in JSONSchema format:\n\n",
                False,
            )
        )
        segments.append(Segment(render_tool_schema(tools), False))
        segments.append(
            Segment(
                "\n\nWhen making tool calls, use XML format to invoke tools and pass parameters:\n\n"
                "<tool_call>\n"
                "<invoke name=\"tool-name-1\">\n"
                "<parameter name=\"param-key-1\">param-value-1</parameter>\n"
                "<parameter name=\"param-key-2\">param-value-2</parameter>\n"
                "...\n"
                "</invoke>\n"
                "</tool_call>",
                False,
            )
        )
    segments.append(Segment(f"{IM_END_TOKEN}\n", False))

    base_offset = 1 if split_system_and_conversation(messages)[0] is not None else 0
    for idx, message in enumerate(conversation_messages, start=base_offset):
        role = message["role"]
        has_next_message = idx - base_offset + 1 < len(conversation_messages)
        suffix = "\n" if has_next_message or add_generation_prompt else ""

        if role == "user":
            segments.append(Segment(f"{IM_START_TOKEN}User: ", False))
            segments.append(Segment(visible_text(message.get("content")), False))
            segments.append(Segment(f"{IM_END_TOKEN}{suffix}", False))
            continue

        if role == "assistant":
            trainable = idx == last_assistant_idx
            segments.append(Segment(ASSISTANT_PREFIX, False))
            content = visible_text(message.get("content"))
            if content:
                segments.append(Segment(content, trainable))
            if message.get("tool_calls"):
                if not content or not content.endswith("\n"):
                    segments.append(Segment("\n", trainable))
                segments.append(Segment(render_tool_calls(message["tool_calls"]), trainable))
            segments.append(Segment(f"{IM_END_TOKEN}{suffix}", trainable))
            continue

        if role == "tool":
            prev_role = conversation_messages[idx - base_offset - 1]["role"] if idx - base_offset > 0 else None
            next_index = idx - base_offset + 1
            next_role = conversation_messages[next_index]["role"] if next_index < len(conversation_messages) else None
            if prev_role != "tool":
                segments.append(Segment(f"{IM_START_TOKEN}Tool: ", False))
            segments.append(Segment("\n" + render_tool_response(message), False))
            if next_role != "tool":
                segments.append(Segment(f"{IM_END_TOKEN}{suffix}", False))
            continue

        raise ValueError(f"Unsupported role for current SFT pipeline: {role!r}")

    if add_generation_prompt:
        prompt = f"{IM_START_TOKEN}Assistant:"
        if enable_thinking:
            prompt += " <think>\n"
        elif no_add_thinking:
            prompt += " "
        else:
            prompt += f" {NO_THINKING_PREFIX}"
        segments.append(Segment(prompt, False))

    return segments


def render_chat_template(
    template,
    messages: Sequence[dict],
    *,
    tools: Sequence[dict] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    no_add_thinking: bool = False,
) -> str:
    compiled = template if hasattr(template, "render") else compile_chat_template(str(template))
    return _render_template(
        compiled,
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        no_add_thinking=no_add_thinking,
    )


def encode_segments(
    tokenizer: TRIE_TOKENIZER,
    segments: Sequence[Segment],
    *,
    append_eod: bool = True,
    eod_trainable: bool = False,
) -> EncodedDocument:
    input_ids: list[int] = []
    loss_mask: list[int] = []
    for segment in segments:
        token_ids = tokenizer.encode(segment.text)
        input_ids.extend(token_ids)
        loss_mask.extend([1 if segment.trainable else 0] * len(token_ids))

    if append_eod:
        eod_ids = tokenizer.encode(EOD_TOKEN)
        input_ids.extend(eod_ids)
        loss_mask.extend([1 if eod_trainable else 0] * len(eod_ids))

    return EncodedDocument(input_ids=input_ids, loss_mask=loss_mask)


def compile_chat_template(template_text: str):
    return _build_jinja_env().from_string(template_text)


def load_chat_template(template_path: str):
    template_text = Path(template_path).read_text(encoding="utf-8")
    return compile_chat_template(template_text)


def _prepare_render_inputs(
    record: dict,
    *,
    current_date: str | None = None,
    current_location: str | None = None,
) -> tuple[list[dict], list[dict] | None]:
    normalized = normalize_record(record)
    messages = trim_trailing_tool_messages(normalized["messages"])
    if current_date is not None or current_location is not None:
        system_message, _ = split_system_and_conversation(messages)
        if system_message is None:
            system_message = {}
        if current_date is not None:
            system_message["current_date"] = current_date
        if current_location is not None:
            system_message["current_location"] = current_location
        if messages and messages[0].get("role") == "system":
            messages = copy.deepcopy(messages)
            messages[0] = system_message
        else:
            messages = [{"role": "system", **system_message}] + copy.deepcopy(messages)
    return messages, normalized.get("tools")


def _render_prefix_and_full(
    record: dict,
    *,
    template,
    current_date: str | None = None,
    current_location: str | None = None,
) -> tuple[str, str]:
    messages, tools = _prepare_render_inputs(
        record,
        current_date=current_date,
        current_location=current_location,
    )
    final_assistant = last_assistant_message(messages)
    if final_assistant is None:
        raise ValueError("SFT record must include at least one assistant message.")

    final_content = visible_text(final_assistant.get("content"))
    has_existing_think = _assistant_has_existing_think(final_content)
    prefix_messages = _messages_before_last_assistant(messages)
    full_messages = _messages_with_normalized_final_assistant(messages)

    prefix_text = _render_template(
        template,
        messages=prefix_messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
        no_add_thinking=has_existing_think,
    )
    full_text = _render_template(
        template,
        messages=full_messages,
        tools=tools,
        add_generation_prompt=False,
        enable_thinking=False,
        no_add_thinking=has_existing_think,
    )
    return prefix_text, full_text


def build_document_from_record(
    record: dict,
    *,
    tokenizer: TRIE_TOKENIZER,
    template,
    current_date: str | None = None,
    current_location: str | None = None,
) -> EncodedDocument:
    prefix_text, full_text = _render_prefix_and_full(
        record,
        template=template,
        current_date=current_date,
        current_location=current_location,
    )
    train_start, train_end = _compute_trainable_span(full_text, prefix_text)
    char_mask = _char_mask_from_span(full_text, train_start, train_end)
    input_ids, char_spans = _tokenize_with_char_spans(tokenizer, full_text)
    loss_mask = _loss_mask_from_char_mask(char_mask, char_spans)

    eod_ids = tokenizer.encode(EOD_TOKEN)
    input_ids.extend(eod_ids)
    loss_mask.extend([1] * len(eod_ids))
    return EncodedDocument(input_ids=input_ids, loss_mask=loss_mask)


def _encode_separator(tokenizer: TRIE_TOKENIZER) -> EncodedDocument:
    token_ids = tokenizer.encode("\n")
    return EncodedDocument(input_ids=token_ids, loss_mask=[0] * len(token_ids))


def pack_encoded_documents(
    documents: Iterable[EncodedDocument],
    *,
    pack_length: int,
    pad_token_id: int,
    separator: EncodedDocument | None = None,
) -> Iterable[EncodedDocument]:
    if pack_length <= 0:
        raise ValueError("pack_length must be a positive integer.")

    packed_ids: list[int] = []
    packed_mask: list[int] = []
    separator = separator or EncodedDocument(input_ids=[], loss_mask=[])

    def flush_padded() -> EncodedDocument | None:
        nonlocal packed_ids, packed_mask
        if not packed_ids:
            return None
        padding = pack_length - len(packed_ids)
        packed = EncodedDocument(
            input_ids=packed_ids + [pad_token_id] * padding,
            loss_mask=packed_mask + [0] * padding,
        )
        packed_ids = []
        packed_mask = []
        return packed

    for document in documents:
        if len(document.input_ids) != len(document.loss_mask):
            raise ValueError("Token ids and loss mask must have identical lengths.")
        if len(document.input_ids) > pack_length:
            raise ValueError(
                f"Document length {len(document.input_ids)} exceeds pack_length {pack_length}."
            )

        next_ids = document.input_ids
        next_mask = document.loss_mask
        if packed_ids and separator.input_ids:
            next_ids = separator.input_ids + next_ids
            next_mask = separator.loss_mask + next_mask

        if len(packed_ids) + len(next_ids) > pack_length:
            packed = flush_padded()
            if packed is not None:
                yield packed
            next_ids = document.input_ids
            next_mask = document.loss_mask

        packed_ids.extend(next_ids)
        packed_mask.extend(next_mask)

        if len(packed_ids) == pack_length:
            packed = flush_padded()
            if packed is not None:
                yield packed

    if packed_ids:
        packed = flush_padded()
        if packed is not None:
            yield packed


def pack_encoded_documents_best_fit_decreasing(
    documents: Iterable[EncodedDocument],
    *,
    pack_length: int,
    pad_token_id: int,
    separator: EncodedDocument | None = None,
) -> Iterable[EncodedDocument]:
    if pack_length <= 0:
        raise ValueError("pack_length must be a positive integer.")

    separator = separator or EncodedDocument(input_ids=[], loss_mask=[])
    source_documents = list(documents)
    for document in source_documents:
        if len(document.input_ids) != len(document.loss_mask):
            raise ValueError("Token ids and loss mask must have identical lengths.")
        if len(document.input_ids) > pack_length:
            raise ValueError(
                f"Document length {len(document.input_ids)} exceeds pack_length {pack_length}."
            )

    sorted_documents = sorted(source_documents, key=lambda document: len(document.input_ids), reverse=True)
    bins: list[tuple[list[int], list[int]]] = []
    remaining_index: list[tuple[int, int]] = []

    for document in sorted_documents:
        append_length = len(document.input_ids)
        if separator.input_ids:
            append_length += len(separator.input_ids)
        position = bisect_left(remaining_index, (append_length, -1))
        best_index = None
        while position < len(remaining_index):
            remaining, candidate_index = remaining_index[position]
            packed_ids, _ = bins[candidate_index]
            actual_append_length = len(document.input_ids)
            if packed_ids and separator.input_ids:
                actual_append_length += len(separator.input_ids)
            if actual_append_length <= remaining:
                best_index = candidate_index
                remaining_index.pop(position)
                break
            position += 1

        if best_index is None:
            bin_index = len(bins)
            bins.append((list(document.input_ids), list(document.loss_mask)))
            remaining = pack_length - len(document.input_ids)
            insort(remaining_index, (remaining, bin_index))
            continue

        packed_ids, packed_mask = bins[best_index]
        if packed_ids and separator.input_ids:
            packed_ids.extend(separator.input_ids)
            packed_mask.extend(separator.loss_mask)
        packed_ids.extend(document.input_ids)
        packed_mask.extend(document.loss_mask)
        remaining = pack_length - len(packed_ids)
        insort(remaining_index, (remaining, best_index))

    for packed_ids, packed_mask in bins:
        padding = pack_length - len(packed_ids)
        yield EncodedDocument(
            input_ids=packed_ids + [pad_token_id] * padding,
            loss_mask=packed_mask + [0] * padding,
        )


def pad_encoded_documents(
    documents: Iterable[EncodedDocument],
    *,
    pad_length: int,
    pad_token_id: int,
) -> Iterable[EncodedDocument]:
    if pad_length <= 0:
        raise ValueError("pad_length must be a positive integer.")

    for document in documents:
        if len(document.input_ids) != len(document.loss_mask):
            raise ValueError("Token ids and loss mask must have identical lengths.")
        if len(document.input_ids) > pad_length:
            continue
        padding = pad_length - len(document.input_ids)
        yield EncodedDocument(
            input_ids=list(document.input_ids) + [pad_token_id] * padding,
            loss_mask=list(document.loss_mask) + [0] * padding,
        )


def load_non_empty_lines(input_path: str) -> list[str]:
    with open(input_path, "r", encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]


def load_non_empty_source_lines(input_path: str) -> list[JsonlSourceLine]:
    source_lines: list[JsonlSourceLine] = []
    with open(input_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if stripped:
                source_lines.append(
                    JsonlSourceLine(
                        text=stripped,
                        source_path=str(Path(input_path)),
                        line_number=line_number,
                    )
                )
    return source_lines


def _emit_read_progress(
    progress_callback: ProgressCallback | None,
    *,
    done: int,
    total: int,
    source_path: str,
    group_index: int | None = None,
    group_count: int | None = None,
) -> None:
    if progress_callback is None:
        return
    event: dict[str, object] = {
        "stage": "read-jsonl",
        "unit": "files",
        "done": done,
        "total": total,
        "source_path": source_path,
    }
    if group_index is not None:
        event["group_index"] = group_index
    if group_count is not None:
        event["group_count"] = group_count
    progress_callback(event)


def _emit_group_progress(
    progress_callback: ProgressCallback | None,
    *,
    stage: str,
    done: int,
    total: int,
    source_path: str,
    group_index: int | None = None,
    group_count: int | None = None,
) -> None:
    if progress_callback is None:
        return
    event: dict[str, object] = {
        "stage": stage,
        "unit": "groups",
        "done": done,
        "total": total,
        "source_path": source_path,
    }
    if group_index is not None:
        event["group_index"] = group_index
    if group_count is not None:
        event["group_count"] = group_count
    progress_callback(event)


def normalize_input_paths(input_jsonl: str | Sequence[str]) -> list[str]:
    if isinstance(input_jsonl, (str, Path)):
        raw_paths = [Path(input_jsonl)]
    else:
        raw_paths = [Path(path) for path in input_jsonl]
    if not raw_paths:
        raise ValueError("At least one input JSONL path is required.")

    paths: list[str] = []
    for raw_path in raw_paths:
        if raw_path.is_dir():
            jsonl_paths = sorted(
                (
                    child
                    for child in raw_path.rglob("*")
                    if child.is_file() and child.suffix.lower() == ".jsonl"
                ),
                key=lambda path: str(path),
            )
            if not jsonl_paths:
                raise ValueError(f"No .jsonl files found in input directory: {raw_path}")
            paths.extend(str(path) for path in jsonl_paths)
        else:
            paths.append(str(raw_path))
    return paths


ProgressCallback = Callable[[dict[str, object]], None]
ErrorCallback = Callable[[dict[str, object]], None]


def load_jsonl_sources(
    input_jsonl: str | Sequence[str],
    *,
    num_workers: int = 1,
    process_pool: ProcessPoolExecutor | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_group_index: int | None = None,
    progress_group_count: int | None = None,
) -> list[JsonlSourceLine]:
    input_paths = normalize_input_paths(input_jsonl)
    total_files = len(input_paths)
    if num_workers <= 1 or len(input_paths) == 1:
        sources: list[JsonlSourceLine] = []
        for done, input_path in enumerate(input_paths, start=1):
            sources.extend(load_non_empty_source_lines(input_path))
            _emit_read_progress(
                progress_callback,
                done=done,
                total=total_files,
                source_path=input_path,
                group_index=progress_group_index,
                group_count=progress_group_count,
            )
        return sources

    executor = process_pool
    close_executor = False
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        close_executor = True
    try:
        grouped_sources = executor.map(load_non_empty_source_lines, input_paths)
        sources: list[JsonlSourceLine] = []
        for done, (input_path, group) in enumerate(zip(input_paths, grouped_sources), start=1):
            sources.extend(group)
            _emit_read_progress(
                progress_callback,
                done=done,
                total=total_files,
                source_path=input_path,
                group_index=progress_group_index,
                group_count=progress_group_count,
            )
        return sources
    finally:
        if close_executor:
            executor.shutdown()


def shuffled_epoch_lines(
    lines: Sequence[str],
    n_epoch: int,
    rng: random.Random,
    *,
    shuffle: bool = True,
) -> list[str]:
    shuffled_lines: list[str] = []
    for _ in range(n_epoch):
        epoch_lines = list(lines)
        if shuffle:
            rng.shuffle(epoch_lines)
        shuffled_lines.extend(epoch_lines)
    return shuffled_lines


def shuffled_epoch_sources(
    sources: Sequence[JsonlSourceLine],
    n_epoch: int,
    rng: random.Random,
    *,
    shuffle: bool = True,
) -> list[JsonlSourceLine]:
    shuffled_sources: list[JsonlSourceLine] = []
    for _ in range(n_epoch):
        epoch_sources = list(sources)
        if shuffle:
            rng.shuffle(epoch_sources)
        shuffled_sources.extend(epoch_sources)
    return shuffled_sources


def _build_document_from_source_line(
    source: JsonlSourceLine,
    *,
    tokenizer: TRIE_TOKENIZER,
    template,
    current_date: str | None = None,
    current_location: str | None = None,
) -> EncodedDocument:
    record = None
    try:
        record = json.loads(source.text)
    except json.JSONDecodeError as exc:
        raise SFTDocumentBuildError(_source_error_event(source, exc)) from exc
    try:
        return build_document_from_record(
            record,
            tokenizer=tokenizer,
            template=template,
            current_date=current_date,
            current_location=current_location,
        )
    except Exception as exc:
        raise SFTDocumentBuildError(_source_error_event(source, exc, record=record)) from exc

_PROCESS_TOKENIZER: TRIE_TOKENIZER | None = None
_PROCESS_TEMPLATE = None
_PROCESS_CURRENT_DATE: str | None = None
_PROCESS_CURRENT_LOCATION: str | None = None


def _init_document_worker(
    vocab_path: str,
    template_path: str,
    current_date: str | None,
    current_location: str | None,
) -> None:
    global _PROCESS_TOKENIZER, _PROCESS_TEMPLATE, _PROCESS_CURRENT_DATE, _PROCESS_CURRENT_LOCATION
    _PROCESS_TOKENIZER = TRIE_TOKENIZER(vocab_path, strict_length=True)
    _PROCESS_TEMPLATE = load_chat_template(template_path)
    _PROCESS_CURRENT_DATE = current_date
    _PROCESS_CURRENT_LOCATION = current_location


def _build_document_from_source_line_in_worker(source: JsonlSourceLine) -> EncodedDocument:
    if _PROCESS_TOKENIZER is None or _PROCESS_TEMPLATE is None:
        raise RuntimeError("SFT document worker was not initialized.")
    return _build_document_from_source_line(
        source,
        tokenizer=_PROCESS_TOKENIZER,
        template=_PROCESS_TEMPLATE,
        current_date=_PROCESS_CURRENT_DATE,
        current_location=_PROCESS_CURRENT_LOCATION,
    )


def _emit_document_progress(
    progress_callback: ProgressCallback | None,
    *,
    done: int,
    total: int | None,
    source: JsonlSourceLine,
    group_index: int | None = None,
    group_count: int | None = None,
) -> None:
    if progress_callback is None:
        return
    event: dict[str, object] = {
        "stage": "render-tokenize",
        "done": done,
        "source_path": source.source_path,
        "line_number": source.line_number,
    }
    if total is not None:
        event["total"] = total
    if group_index is not None:
        event["group_index"] = group_index
    if group_count is not None:
        event["group_count"] = group_count
    progress_callback(event)


def build_documents_from_sources(
    sources: Sequence[JsonlSourceLine],
    *,
    tokenizer: TRIE_TOKENIZER | None = None,
    template=None,
    vocab_path: str | None = None,
    template_path: str | None = None,
    current_date: str | None = None,
    current_location: str | None = None,
    num_workers: int = 1,
    worker_chunksize: int = 64,
    process_pool: ProcessPoolExecutor | None = None,
    progress_callback: ProgressCallback | None = None,
    error_callback: ErrorCallback | None = None,
    progress_total: int | None = None,
    progress_group_index: int | None = None,
    progress_group_count: int | None = None,
) -> Iterable[EncodedDocument]:
    if worker_chunksize <= 0:
        raise ValueError("worker_chunksize must be a positive integer.")
    if num_workers <= 1:
        if tokenizer is None:
            if vocab_path is None:
                raise ValueError("vocab_path is required when tokenizer is not provided.")
            tokenizer = TRIE_TOKENIZER(vocab_path, strict_length=True)
        if template is None:
            if template_path is None:
                raise ValueError("template_path is required when template is not provided.")
            template = load_chat_template(template_path)

    if num_workers <= 1:
        done = 0
        for source in sources:
            try:
                document = _build_document_from_source_line(
                    source,
                    tokenizer=tokenizer,
                    template=template,
                    current_date=current_date,
                    current_location=current_location,
                )
            except Exception as exc:
                _emit_exception_error(error_callback, exc, source=source)
                raise
            done += 1
            _emit_document_progress(
                progress_callback,
                done=done,
                total=progress_total,
                source=source,
                group_index=progress_group_index,
                group_count=progress_group_count,
            )
            yield document
        return

    if vocab_path is None or template_path is None:
        raise ValueError("vocab_path and template_path are required when num_workers > 1.")

    executor = process_pool
    close_executor = False
    if executor is None:
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_document_worker,
            initargs=(vocab_path, template_path, current_date, current_location),
        )
        close_executor = True
    try:
        done = 0
        mapped_documents = executor.map(
            _build_document_from_source_line_in_worker,
            sources,
            chunksize=worker_chunksize,
        )
        try:
            for source, document in zip(sources, mapped_documents):
                done += 1
                _emit_document_progress(
                    progress_callback,
                    done=done,
                    total=progress_total,
                    source=source,
                    group_index=progress_group_index,
                    group_count=progress_group_count,
                )
                yield document
        except Exception as exc:
            _emit_exception_error(error_callback, exc)
            raise
    finally:
        if close_executor:
            executor.shutdown()


def append_documents_to_builders(
    token_builder: MMapIndexedDatasetBuilder,
    mask_builder: MMapIndexedDatasetBuilder,
    documents: Iterable[EncodedDocument],
    *,
    token_dtype=np.uint16,
    mask_dtype=np.uint8,
):
    doc_count = 0
    token_count = 0
    trainable_count = 0
    for document in documents:
        if len(document.input_ids) != len(document.loss_mask):
            raise ValueError("Token ids and loss mask must have identical lengths.")
        token_arr = np.array(document.input_ids, dtype=token_dtype)
        mask_arr = np.array(document.loss_mask, dtype=mask_dtype)
        token_builder.add_item(token_arr)
        token_builder.end_document()
        mask_builder.add_item(mask_arr)
        mask_builder.end_document()
        doc_count += 1
        token_count += int(token_arr.size)
        trainable_count += int(mask_arr.sum())

    return {
        "documents": doc_count,
        "tokens": token_count,
        "trainable_tokens": trainable_count,
    }


def write_documents(
    output_prefix: str,
    documents: Iterable[EncodedDocument],
    *,
    token_dtype=np.uint16,
    mask_dtype=np.uint8,
):
    token_builder = MMapIndexedDatasetBuilder(data_file_path(output_prefix), dtype=token_dtype)
    mask_prefix = mask_prefix_path(output_prefix)
    mask_builder = MMapIndexedDatasetBuilder(data_file_path(mask_prefix), dtype=mask_dtype)

    stats = append_documents_to_builders(
        token_builder,
        mask_builder,
        documents,
        token_dtype=token_dtype,
        mask_dtype=mask_dtype,
    )

    token_builder.finalize(index_file_path(output_prefix))
    mask_builder.finalize(index_file_path(mask_prefix))
    stats["mask_path"] = mask_file_path(output_prefix)
    return stats


def collect_filtered_documents(
    documents: Iterable[EncodedDocument],
    *,
    max_length: int | None,
) -> tuple[list[EncodedDocument], int]:
    kept: list[EncodedDocument] = []
    filtered = 0
    for document in documents:
        if len(document.input_ids) != len(document.loss_mask):
            raise ValueError("Token ids and loss mask must have identical lengths.")
        if max_length is not None and len(document.input_ids) > max_length:
            filtered += 1
            continue
        kept.append(document)
    return kept, filtered


def iter_filtered_documents(
    documents: Iterable[EncodedDocument],
    *,
    max_length: int | None,
    stats: FilterStats | None = None,
) -> Iterable[EncodedDocument]:
    for document in documents:
        if len(document.input_ids) != len(document.loss_mask):
            raise ValueError("Token ids and loss mask must have identical lengths.")
        if max_length is not None and len(document.input_ids) > max_length:
            if stats is not None:
                stats.filtered += 1
            continue
        yield document


def _binidx_pair_exists(prefix: str) -> bool:
    return (
        Path(data_file_path(prefix)).is_file()
        and Path(index_file_path(prefix)).is_file()
        and Path(data_file_path(mask_prefix_path(prefix))).is_file()
        and Path(index_file_path(mask_prefix_path(prefix))).is_file()
    )


def _cache_meta_path(prefix: str) -> Path:
    return Path(prefix + ".meta.json")


def _load_cache_meta(prefix: str) -> dict[str, int] | None:
    meta_path = _cache_meta_path(prefix)
    if not _binidx_pair_exists(prefix) or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required_keys = {"source_lines", "source_documents", "filtered_documents"}
    if not required_keys.issubset(meta):
        return None
    return {key: int(value) for key, value in meta.items() if isinstance(value, int)}


def _write_cache_meta(prefix: str, meta: dict[str, int]) -> None:
    meta_path = _cache_meta_path(prefix)
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, meta_path)


def _merge_binidx_pair_into_builders(
    prefix: str,
    token_builder: MMapIndexedDatasetBuilder,
    mask_builder: MMapIndexedDatasetBuilder,
    *,
    token_dtype=np.uint16,
    mask_dtype=np.uint8,
) -> dict[str, int]:
    token_dataset = MMapIndexedDataset(prefix)
    mask_dataset = MMapIndexedDataset(mask_prefix_path(prefix))
    if len(token_dataset) != len(mask_dataset):
        raise ValueError(f"Cached token and mask document count mismatch for {prefix}.")
    if not np.array_equal(token_dataset.sizes, mask_dataset.sizes):
        raise ValueError(f"Cached token and mask document sizes differ for {prefix}.")

    stats = {"documents": 0, "tokens": 0, "trainable_tokens": 0}
    for index in range(len(token_dataset)):
        token_arr = np.asarray(token_dataset[index], dtype=token_dtype)
        mask_arr = np.asarray(mask_dataset[index], dtype=mask_dtype)
        token_builder.add_item(token_arr)
        token_builder.end_document()
        mask_builder.add_item(mask_arr)
        mask_builder.end_document()
        stats["documents"] += 1
        stats["tokens"] += int(token_arr.size)
        stats["trainable_tokens"] += int(mask_arr.sum())
    return stats


def _file_signature(path: str) -> dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_group_prefix(
    cache_dir: str,
    *,
    group_index: int,
    input_group: Sequence[str],
    cache_config: dict[str, object],
) -> str:
    digest_payload = {
        "config": cache_config,
        "files": [_file_signature(path) for path in input_group],
    }
    digest = hashlib.sha1(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return str(Path(cache_dir) / f"group_{group_index:06d}_{digest}")


def write_best_fit_decreasing_sharded_documents(
    output_prefix: str,
    input_paths: Sequence[str],
    *,
    vocab_path: str,
    template_path: str,
    n_epoch: int,
    seed: int,
    pack_length: int,
    current_date: str | None = None,
    current_location: str | None = None,
    num_workers: int = 1,
    shuffle: bool = True,
    pack_shard_group_size: int = 1,
    worker_chunksize: int = 64,
    pack_cache_dir: str | None = None,
    token_dtype=np.uint16,
    mask_dtype=np.uint8,
    progress_callback: ProgressCallback | None = None,
    error_callback: ErrorCallback | None = None,
):
    if pack_shard_group_size <= 0:
        raise ValueError("pack_shard_group_size must be a positive integer.")

    if worker_chunksize <= 0:
        raise ValueError("worker_chunksize must be a positive integer.")
    if pack_cache_dir is not None and shuffle:
        raise ValueError("pack_cache_dir requires shuffle=False for deterministic cache reuse.")

    token_builder = MMapIndexedDatasetBuilder(data_file_path(output_prefix), dtype=token_dtype)
    mask_prefix = mask_prefix_path(output_prefix)
    mask_builder = MMapIndexedDatasetBuilder(data_file_path(mask_prefix), dtype=mask_dtype)
    rng = random.Random(seed)
    tokenizer = TRIE_TOKENIZER(vocab_path, strict_length=True)
    pad_token_id = eod_token_id(tokenizer)
    separator = _encode_separator(tokenizer)
    if pack_cache_dir is not None:
        Path(pack_cache_dir).mkdir(parents=True, exist_ok=True)
    cache_config = {
        "vocab": _file_signature(vocab_path),
        "template": _file_signature(template_path),
        "n_epoch": n_epoch,
        "seed": seed,
        "pack_length": pack_length,
        "current_date": current_date,
        "current_location": current_location,
        "shuffle": shuffle,
        "pack_shard_group_size": pack_shard_group_size,
    }

    stats = {
        "documents": 0,
        "tokens": 0,
        "trainable_tokens": 0,
        "source_lines": 0,
        "source_documents": 0,
        "filtered_documents": 0,
    }

    read_pool = (
        ProcessPoolExecutor(max_workers=num_workers)
        if num_workers > 1 and len(input_paths) > 1
        else None
    )
    document_pool = None
    try:
        group_count = (len(input_paths) + pack_shard_group_size - 1) // pack_shard_group_size
        for group_index, start in enumerate(range(0, len(input_paths), pack_shard_group_size), start=1):
            input_group = input_paths[start:start + pack_shard_group_size]
            cache_prefix = (
                _cache_group_prefix(
                    pack_cache_dir,
                    group_index=group_index,
                    input_group=input_group,
                    cache_config=cache_config,
                )
                if pack_cache_dir is not None
                else None
            )
            cache_meta = _load_cache_meta(cache_prefix) if cache_prefix is not None else None
            if cache_prefix is not None and cache_meta is not None:
                shard_stats = _merge_binidx_pair_into_builders(
                    cache_prefix,
                    token_builder,
                    mask_builder,
                    token_dtype=token_dtype,
                    mask_dtype=mask_dtype,
                )
                stats["documents"] += shard_stats["documents"]
                stats["tokens"] += shard_stats["tokens"]
                stats["trainable_tokens"] += shard_stats["trainable_tokens"]
                stats["source_lines"] += cache_meta["source_lines"]
                stats["source_documents"] += cache_meta["source_documents"]
                stats["filtered_documents"] += cache_meta["filtered_documents"]
                _emit_group_progress(
                    progress_callback,
                    stage="cache-hit",
                    done=group_index,
                    total=group_count,
                    source_path=cache_prefix,
                    group_index=group_index,
                    group_count=group_count,
                )
                continue

            sources = load_jsonl_sources(
                input_group,
                num_workers=num_workers,
                process_pool=read_pool,
                progress_callback=progress_callback,
                progress_group_index=group_index,
                progress_group_count=group_count,
            )
            shuffled_sources = shuffled_epoch_sources(sources, n_epoch, rng, shuffle=shuffle)
            if document_pool is None and num_workers > 1:
                document_pool = ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_document_worker,
                    initargs=(vocab_path, template_path, current_date, current_location),
                )
            documents = build_documents_from_sources(
                shuffled_sources,
                vocab_path=vocab_path,
                template_path=template_path,
                current_date=current_date,
                current_location=current_location,
                num_workers=num_workers,
                worker_chunksize=worker_chunksize,
                process_pool=document_pool,
                progress_callback=progress_callback,
                error_callback=error_callback,
                progress_total=len(shuffled_sources),
                progress_group_index=group_index,
                progress_group_count=group_count,
            )
            filter_stats = FilterStats()
            documents = iter_filtered_documents(
                documents,
                max_length=pack_length,
                stats=filter_stats,
            )
            packed_documents = pack_encoded_documents_best_fit_decreasing(
                documents,
                pack_length=pack_length,
                pad_token_id=pad_token_id,
                separator=separator,
            )
            if cache_prefix is not None:
                shard_stats = write_documents(
                    cache_prefix,
                    packed_documents,
                    token_dtype=token_dtype,
                    mask_dtype=mask_dtype,
                )
                _write_cache_meta(
                    cache_prefix,
                    {
                        "source_lines": len(sources),
                        "source_documents": len(shuffled_sources),
                        "filtered_documents": filter_stats.filtered,
                    },
                )
                shard_stats = _merge_binidx_pair_into_builders(
                    cache_prefix,
                    token_builder,
                    mask_builder,
                    token_dtype=token_dtype,
                    mask_dtype=mask_dtype,
                )
                _emit_group_progress(
                    progress_callback,
                    stage="cache-write",
                    done=group_index,
                    total=group_count,
                    source_path=cache_prefix,
                    group_index=group_index,
                    group_count=group_count,
                )
            else:
                shard_stats = append_documents_to_builders(
                    token_builder,
                    mask_builder,
                    packed_documents,
                    token_dtype=token_dtype,
                    mask_dtype=mask_dtype,
                )

            stats["documents"] += shard_stats["documents"]
            stats["tokens"] += shard_stats["tokens"]
            stats["trainable_tokens"] += shard_stats["trainable_tokens"]
            stats["source_lines"] += len(sources)
            stats["source_documents"] += len(shuffled_sources)
            stats["filtered_documents"] += filter_stats.filtered
    finally:
        if read_pool is not None:
            read_pool.shutdown()
        if document_pool is not None:
            document_pool.shutdown()

    token_builder.finalize(index_file_path(output_prefix))
    mask_builder.finalize(index_file_path(mask_prefix))
    stats["mask_path"] = mask_file_path(output_prefix)
    return stats


def default_output_prefix(input_jsonl: str | Sequence[str]) -> str:
    if isinstance(input_jsonl, (str, Path)):
        input_paths = [Path(input_jsonl)]
    else:
        input_paths = [Path(path) for path in input_jsonl]
    if len(input_paths) != 1:
        raise ValueError("--out-prefix is required when building from multiple input JSONL paths.")
    input_path = input_paths[0].resolve()
    if input_path.is_dir():
        return str(input_path)
    return str(input_path.with_suffix(""))


def build_binidx_dataset(
    input_jsonl: str | Sequence[str],
    *,
    output_prefix: str | None,
    vocab_path: str,
    template_path: str,
    n_epoch: int,
    seed: int,
    pack_length: int | None = None,
    pad_length: int | None = None,
    current_date: str | None = None,
    current_location: str | None = None,
    num_workers: int = 1,
    shuffle: bool = True,
    pack_strategy: str = "ordered",
    pack_shard_group_size: int = 1,
    worker_chunksize: int = 64,
    pack_cache_dir: str | None = None,
    progress_callback: ProgressCallback | None = None,
    error_callback: ErrorCallback | None = None,
):
    if num_workers <= 0:
        raise ValueError("num_workers must be a positive integer.")
    if pack_shard_group_size <= 0:
        raise ValueError("pack_shard_group_size must be a positive integer.")
    if pack_length is not None and pad_length is not None:
        raise ValueError("pack_length and pad_length are mutually exclusive.")
    if pack_strategy not in {"ordered", "best-fit-decreasing"}:
        raise ValueError("pack_strategy must be 'ordered' or 'best-fit-decreasing'.")
    if worker_chunksize <= 0:
        raise ValueError("worker_chunksize must be a positive integer.")
    if pack_cache_dir is not None and not (pack_length is not None and pack_strategy == "best-fit-decreasing"):
        raise ValueError("pack_cache_dir is only supported with best-fit-decreasing packing.")
    if pack_cache_dir is not None and shuffle:
        raise ValueError("pack_cache_dir requires shuffle=False for deterministic cache reuse.")

    input_paths = normalize_input_paths(input_jsonl)
    prefix = output_prefix or default_output_prefix(input_jsonl)

    if pack_length is not None and pack_strategy == "best-fit-decreasing":
        stats = write_best_fit_decreasing_sharded_documents(
            prefix,
            input_paths,
            vocab_path=vocab_path,
            template_path=template_path,
            n_epoch=n_epoch,
            seed=seed,
            pack_length=pack_length,
            current_date=current_date,
            current_location=current_location,
            num_workers=num_workers,
            shuffle=shuffle,
            pack_shard_group_size=pack_shard_group_size,
            worker_chunksize=worker_chunksize,
            pack_cache_dir=pack_cache_dir,
            progress_callback=progress_callback,
            error_callback=error_callback,
        )
    else:
        tokenizer = TRIE_TOKENIZER(vocab_path, strict_length=True)
        sources = load_jsonl_sources(input_paths, num_workers=num_workers, progress_callback=progress_callback)
        rng = random.Random(seed)
        shuffled_sources = shuffled_epoch_sources(sources, n_epoch, rng, shuffle=shuffle)
        documents = build_documents_from_sources(
            shuffled_sources,
            vocab_path=vocab_path,
            template_path=template_path,
            current_date=current_date,
            current_location=current_location,
            num_workers=num_workers,
            worker_chunksize=worker_chunksize,
            progress_callback=progress_callback,
            error_callback=error_callback,
            progress_total=len(shuffled_sources),
        )

        max_source_length = pack_length if pack_length is not None else pad_length
        documents, filtered_documents = collect_filtered_documents(
            documents,
            max_length=max_source_length,
        )
        if pack_length is not None:
            documents = pack_encoded_documents(
                documents,
                pack_length=pack_length,
                pad_token_id=eod_token_id(tokenizer),
                separator=_encode_separator(tokenizer),
            )
        elif pad_length is not None:
            documents = pad_encoded_documents(
                documents,
                pad_length=pad_length,
                pad_token_id=eod_token_id(tokenizer),
            )

        stats = write_documents(prefix, documents)
        stats["source_lines"] = len(sources)
        stats["source_documents"] = len(shuffled_sources)
        stats["filtered_documents"] = filtered_documents

    stats["output_prefix"] = prefix
    stats["source_files"] = len(input_paths)
    stats["epochs"] = n_epoch
    stats["pack_length"] = pack_length
    stats["pad_length"] = pad_length
    stats["num_workers"] = num_workers
    stats["shuffle"] = shuffle
    stats["pack_strategy"] = pack_strategy
    stats["pack_shard_group_size"] = pack_shard_group_size
    stats["worker_chunksize"] = worker_chunksize
    stats["pack_cache_dir"] = pack_cache_dir
    return stats

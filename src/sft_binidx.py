from __future__ import annotations

import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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

    text_len = len(text)
    byte_to_char = [0] * (len(encoded) + 1)
    byte_index = 0
    for char_index, char in enumerate(text):
        char_bytes = char.encode("utf-8")
        for _ in range(len(char_bytes)):
            byte_to_char[byte_index] = char_index
            byte_index += 1
    byte_to_char[len(encoded)] = text_len

    char_spans = [(byte_to_char[start], byte_to_char[end]) for start, end in byte_spans]
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
    messages = normalized["messages"]
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


def load_jsonl_sources(input_jsonl: str | Sequence[str], *, num_workers: int = 1) -> list[JsonlSourceLine]:
    input_paths = normalize_input_paths(input_jsonl)
    if num_workers <= 1 or len(input_paths) == 1:
        sources: list[JsonlSourceLine] = []
        for input_path in input_paths:
            sources.extend(load_non_empty_source_lines(input_path))
        return sources

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        grouped_sources = executor.map(load_non_empty_source_lines, input_paths)
        return [source for group in grouped_sources for source in group]


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
    try:
        record = json.loads(source.text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {source.source_path}:{source.line_number}: {exc.msg}"
        ) from exc
    return build_document_from_record(
        record,
        tokenizer=tokenizer,
        template=template,
        current_date=current_date,
        current_location=current_location,
    )


def build_documents_from_sources(
    sources: Sequence[JsonlSourceLine],
    *,
    tokenizer: TRIE_TOKENIZER,
    template,
    current_date: str | None = None,
    current_location: str | None = None,
    num_workers: int = 1,
) -> Iterable[EncodedDocument]:
    if num_workers <= 1:
        for source in sources:
            yield _build_document_from_source_line(
                source,
                tokenizer=tokenizer,
                template=template,
                current_date=current_date,
                current_location=current_location,
            )
        return

    def build(source: JsonlSourceLine) -> EncodedDocument:
        return _build_document_from_source_line(
            source,
            tokenizer=tokenizer,
            template=template,
            current_date=current_date,
            current_location=current_location,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(build, sources)


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

    token_builder.finalize(index_file_path(output_prefix))
    mask_builder.finalize(index_file_path(mask_prefix))
    return {
        "documents": doc_count,
        "tokens": token_count,
        "trainable_tokens": trainable_count,
        "mask_path": mask_file_path(output_prefix),
    }


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
):
    if num_workers <= 0:
        raise ValueError("num_workers must be a positive integer.")
    if pack_length is not None and pad_length is not None:
        raise ValueError("pack_length and pad_length are mutually exclusive.")

    template = load_chat_template(template_path)
    tokenizer = TRIE_TOKENIZER(vocab_path, strict_length=True)
    sources = load_jsonl_sources(input_jsonl, num_workers=num_workers)
    rng = random.Random(seed)
    shuffled_sources = shuffled_epoch_sources(sources, n_epoch, rng, shuffle=shuffle)
    prefix = output_prefix or default_output_prefix(input_jsonl)

    documents = build_documents_from_sources(
        shuffled_sources,
        tokenizer=tokenizer,
        template=template,
        current_date=current_date,
        current_location=current_location,
        num_workers=num_workers,
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
    stats["output_prefix"] = prefix
    stats["source_files"] = len(normalize_input_paths(input_jsonl))
    stats["source_lines"] = len(sources)
    stats["source_documents"] = len(shuffled_sources)
    stats["filtered_documents"] = filtered_documents
    stats["epochs"] = n_epoch
    stats["pack_length"] = pack_length
    stats["pad_length"] = pad_length
    stats["num_workers"] = num_workers
    stats["shuffle"] = shuffle
    return stats

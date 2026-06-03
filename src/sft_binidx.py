from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from src.binidx import MMapIndexedDataset


IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
EOD_TOKEN = "<|endoftext|>"
TOOL_CALL_BEGIN = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


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
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return parsed
    if arguments is None:
        return {}
    return arguments


def render_tool_schema(tools: Sequence[dict]) -> str:
    lines = ["<tools>"]
    for tool in tools:
        payload = tool.get("function", tool)
        lines.append(f"<tool>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool>")
    lines.append("</tools>")
    return "\n".join(lines)


def render_tool_calls(tool_calls: Sequence[dict]) -> str:
    parts = [TOOL_CALL_BEGIN]
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
    parts.append(TOOL_CALL_END)
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


def build_template_segments(
    messages: Sequence[dict],
    *,
    tools: Sequence[dict] | None = None,
    current_date: str | None = None,
    current_location: str | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> list[Segment]:
    system_message, conversation_messages = split_system_and_conversation(messages)
    last_assistant_idx = last_assistant_content_index(messages)

    segments: list[Segment] = []
    segments.append(Segment(f"{IM_START_TOKEN}System: ", False))
    segments.append(
        Segment(
            build_system_text(
                system_message,
                current_date=current_date,
                current_location=current_location,
            ),
            False,
        )
    )
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
                f"{TOOL_CALL_BEGIN}\n"
                "<invoke name=\"tool-name-1\">\n"
                "<parameter name=\"param-key-1\">param-value-1</parameter>\n"
                "<parameter name=\"param-key-2\">param-value-2</parameter>\n"
                "...\n"
                "</invoke>\n"
                f"{TOOL_CALL_END}",
                False,
            )
        )
    segments.append(Segment(f"{IM_END_TOKEN}\n", False))

    base_offset = 1 if system_message is not None else 0
    for idx, message in enumerate(conversation_messages, start=base_offset):
        role = message["role"]
        conversation_index = idx - base_offset
        has_next_message = conversation_index + 1 < len(conversation_messages)
        suffix = "\n" if has_next_message or add_generation_prompt else ""

        if role == "user":
            segments.append(Segment(f"{IM_START_TOKEN}User: ", False))
            segments.append(Segment(visible_text(message.get("content")), False))
            segments.append(Segment(f"{IM_END_TOKEN}{suffix}", False))
            continue

        if role == "assistant":
            trainable = idx == last_assistant_idx
            segments.append(Segment(f"{IM_START_TOKEN}Assistant: ", False))
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
            next_index = conversation_index + 1
            next_role = (
                conversation_messages[next_index]["role"]
                if next_index < len(conversation_messages)
                else None
            )
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
            prompt += " <think>"
        else:
            prompt += " <think>\n</think>"
        segments.append(Segment(prompt, False))

    return segments


def render_chat_template(
    template,
    messages: Sequence[dict],
    *,
    tools: Sequence[dict] | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    current_date: str | None = None,
    current_location: str | None = None,
) -> str:
    del template
    return "".join(
        segment.text
        for segment in build_template_segments(
            messages,
            tools=tools,
            current_date=current_date,
            current_location=current_location,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
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


def build_document_from_record(
    record: dict,
    *,
    tokenizer: TRIE_TOKENIZER,
    template,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    current_date: str | None = None,
    current_location: str | None = None,
) -> EncodedDocument:
    del template
    messages = record["messages"]
    tools = record.get("tools")
    last_assistant_idx = last_assistant_content_index(messages)
    segments = build_template_segments(
        messages,
        tools=tools,
        current_date=current_date,
        current_location=current_location,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    return encode_segments(
        tokenizer,
        segments,
        eod_trainable=last_assistant_idx is not None,
    )


def load_chat_template(template_path: str):
    return Path(template_path).read_text(encoding="utf-8")


def load_non_empty_lines(input_path: str) -> list[str]:
    with open(input_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def shuffled_epoch_lines(lines: Sequence[str], n_epoch: int, rng: random.Random) -> list[str]:
    shuffled_lines: list[str] = []
    for _ in range(n_epoch):
        epoch_lines = list(lines)
        rng.shuffle(epoch_lines)
        shuffled_lines.extend(epoch_lines)
    return shuffled_lines


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


def default_output_prefix(input_jsonl: str) -> str:
    input_path = Path(input_jsonl).resolve()
    return str(input_path.with_suffix(""))


def build_binidx_dataset(
    input_jsonl: str,
    *,
    output_prefix: str | None,
    vocab_path: str,
    template_path: str,
    n_epoch: int,
    seed: int,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    current_date: str | None = None,
    current_location: str | None = None,
):
    template = load_chat_template(template_path)
    tokenizer = TRIE_TOKENIZER(vocab_path, strict_length=True)
    lines = load_non_empty_lines(input_jsonl)
    rng = random.Random(seed)
    shuffled_lines = shuffled_epoch_lines(lines, n_epoch, rng)
    prefix = output_prefix or default_output_prefix(input_jsonl)

    documents: list[EncodedDocument] = []
    for line in shuffled_lines:
        record = json.loads(line)
        documents.append(
            build_document_from_record(
                record,
                tokenizer=tokenizer,
                template=template,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                current_date=current_date,
                current_location=current_location,
            )
        )

    stats = write_documents(prefix, documents)
    stats["output_prefix"] = prefix
    stats["source_lines"] = len(lines)
    stats["epochs"] = n_epoch
    return stats

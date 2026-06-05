import io
import copy
import json
import random
import warnings
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.make_sft_binidx import build_arg_parser, main as make_sft_binidx_main
from data.tokenizer.rwkv_tokenizer import TRIE, TRIE_TOKENIZER, parse_vocab_line
from src.binidx import MMapIndexedDataset
from src.sft_binidx import (
    ASSISTANT_PREFIX,
    EOD_TOKEN,
    EncodedDocument,
    IM_START_TOKEN,
    JsonlSourceLine,
    NO_THINKING_PREFIX,
    Segment,
    _assistant_has_existing_think,
    _build_jinja_env,
    _char_mask_from_span,
    _compute_trainable_span,
    _encode_separator,
    _loss_mask_from_char_mask,
    _messages_before_last_assistant,
    _messages_with_normalized_final_assistant,
    _prepare_render_inputs,
    _render_prefix_and_full,
    _tokenize_with_char_spans,
    build_binidx_dataset,
    build_documents_from_sources,
    build_document_from_record,
    build_system_text,
    build_template_segments,
    compile_chat_template,
    data_file_path,
    default_output_prefix,
    eod_token_id,
    encode_segments,
    index_file_path,
    last_assistant_content_index,
    last_assistant_message,
    load_chat_template,
    load_jsonl_sources,
    load_non_empty_lines,
    load_non_empty_source_lines,
    mask_file_path,
    mask_prefix_path,
    normalize_input_paths,
    normalize_tool_calls,
    normalize_record,
    pack_encoded_documents,
    parse_tool_arguments,
    render_chat_template,
    render_tool_calls,
    render_tool_response,
    render_tool_schema,
    shuffled_epoch_lines,
    shuffled_epoch_sources,
    split_system_and_conversation,
    visible_text,
    write_documents,
)


VOCAB_PATH = ROOT / "rwkv_vocab_v20260603.txt"
TEMPLATE_PATH = ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"
TOOLS_JSONL = ROOT / "data" / "SFT" / "tools.jsonl"
MY_SAMPLE_PATH = ROOT / "data" / "SFT" / "sample" / "my_sample.jsonl"
MY_SAMPLE_OUTPUT = ROOT / "data" / "SFT" / "sample" / "my_sample_outs.txt"
THINK_SAMPLE_PATH = ROOT / "data" / "SFT" / "sample" / "think.jsonl"
THINK_SAMPLE_OUTPUT = ROOT / "data" / "SFT" / "sample" / "think_out.txt"
FINAL_PATTERN_PATH = ROOT / "data" / "SFT" / "sample" / "final_pattern.txt"
OLD_VOCAB_PATH = ROOT / "data" / "tokenizer" / "rwkv_vocab_v20230424.txt"


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8")
    return text.replace("\r\n", "\n")


def final_pattern_chunks() -> tuple[str, str, int]:
    text = read_text_auto(FINAL_PATTERN_PATH)
    pad_marker = EOD_TOKEN
    pad_count = 0
    while text.endswith(pad_marker):
        pad_count += 1
        text = text[: -len(pad_marker)]

    boundary = EOD_TOKEN + "\n"
    first, second = text.split(boundary, 1)
    return first + EOD_TOKEN, second, pad_count


def final_pattern_records() -> tuple[dict, dict]:
    no_think_record = json.loads(MY_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])
    think_source = json.loads(THINK_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])
    think_record = copy.deepcopy(no_think_record)
    think_record["messages"][-1]["content"] = think_source["messages"][-1]["content"]
    return think_record, no_think_record


@pytest.fixture(scope="module")
def tokenizer():
    return TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)


@pytest.fixture(scope="module")
def chat_template():
    return load_chat_template(str(TEMPLATE_PATH))


@pytest.fixture(scope="module")
def my_sample_record():
    return json.loads(MY_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])


@pytest.fixture(scope="module")
def think_sample_record():
    return json.loads(THINK_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])


def masked_text(tokenizer: TRIE_TOKENIZER, encoded: EncodedDocument) -> str:
    kept = [token_id for token_id, mask in zip(encoded.input_ids, encoded.loss_mask) if mask == 1]
    return tokenizer.decode(kept)


def test_parse_vocab_line_accepts_new_vocab_spacing_mismatches():
    idx, token_bytes, declared_length = parse_vocab_line("19231 '￼' 3")
    assert idx == 19231
    assert token_bytes == "￼".encode("utf-8")
    assert declared_length == 3


def test_parse_vocab_line_strict_length_raises():
    with pytest.raises(ValueError, match="Token length mismatch"):
        parse_vocab_line("1 'a' 2", strict_length=True)


def test_parse_vocab_line_rejects_non_byte_tokens():
    with pytest.raises(TypeError, match="Unsupported token value type"):
        parse_vocab_line("7 123 3")


def test_new_vocab_loads_special_tokens_without_len_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)
    assert loaded.token2idx[b"<|im_start|>"] == 65530
    assert loaded.token2idx[b"<|im_end|>"] == 65531
    assert loaded.token2idx[b"<|endoftext|>"] == 65532
    assert not [
        warning for warning in caught if "declared byte lengths" in str(warning.message)
    ]


def test_new_vocab_matches_old_vocab_for_fixed_special_tokens():
    target_ids = [19231, 19232, 23247, 43902, 58648, 64156, 64749]
    new_tokenizer = TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)
    old_tokenizer = TRIE_TOKENIZER(str(OLD_VOCAB_PATH), strict_length=True)
    for token_id in target_ids:
        assert new_tokenizer.idx2token[token_id] == old_tokenizer.idx2token[token_id]


def test_tokenizer_encode_decode_and_print_tokens(tokenizer: TRIE_TOKENIZER):
    tokens = tokenizer.encode("Assistant")
    assert tokenizer.decode(tokens) == "Assistant"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        tokenizer.printTokens(tokens[:2])
    assert stdout.getvalue().strip()


def test_trie_repr_and_default_value_storage():
    trie = TRIE()
    leaf = trie.add(b"ab")
    assert b"ab" in leaf.values
    trie_repr = repr(leaf)
    assert "<TRIE" in trie_repr
    assert "97" in trie_repr
    assert "98" in trie_repr


def test_tokenizer_handles_invalid_utf8_decode_and_print(tmp_path):
    vocab_path = tmp_path / "invalid_vocab.txt"
    vocab_path.write_text("0 b'\\xff' 1\n1 'a' 1\n", encoding="utf-8")
    tokenizer = TRIE_TOKENIZER(str(vocab_path), strict_length=True)

    assert tokenizer.decode([0]) == "\ufffd"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        tokenizer.printTokens([0, 1])
    printed = stdout.getvalue()
    assert "0" in printed
    assert "1" in printed


def test_visible_text_supports_string_list_dict_and_none():
    assert visible_text("abc") == "abc"
    assert visible_text(["a", {"type": "text", "text": "b"}, {"text": "c"}, 1, None]) == "abc1"
    assert visible_text(None) == ""
    assert visible_text({"x": 1}) == "{'x': 1}"


def test_build_system_text_uses_message_date_location_and_defaults():
    assert build_system_text(None) == "You are a helpful assistant. Your name is xiaoke and is built by CETC."

    built = build_system_text(
        {
            "content": "系统提示",
            "current_date": "2026-06-03",
            "current_location": "Shanghai, China",
        }
    )
    assert built == "系统提示\nCurrent date: 2026-06-03\nCurrent location: Shanghai, China"

    overridden = build_system_text(
        {"content": "系统提示", "current_date": "old", "current_location": "old"},
        current_date="2026-06-04",
        current_location="Beijing, China",
    )
    assert overridden == "系统提示\nCurrent date: 2026-06-04\nCurrent location: Beijing, China"


def test_parse_tool_arguments_supports_json_xml_empty_and_passthrough():
    assert parse_tool_arguments('{"city":"Shanghai"}') == {"city": "Shanghai"}
    raw = '<parameter name="city">Shanghai</parameter>'
    assert parse_tool_arguments(raw) == raw
    assert parse_tool_arguments("not-json") == "not-json"
    assert parse_tool_arguments("   ") == {}
    source = {"city": "Shanghai"}
    assert parse_tool_arguments(source) is source
    assert parse_tool_arguments(None) == {}


def test_normalize_tool_calls_accepts_empty_input():
    assert normalize_tool_calls(None) is None
    assert normalize_tool_calls([]) == []


def test_normalize_record_converts_json_tool_call_arguments():
    record = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "demo", "arguments": '{"city":"上海"}'}}],
            }
        ]
    }
    normalized = normalize_record(record)
    assert normalized["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"city": "上海"}
    assert record["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"city":"上海"}'


def test_render_tool_schema_and_calls_match_expected_xml_shape():
    tools = [
        {
            "function": {
                "name": "get_weather",
                "description": "desc",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        }
    ]
    schema = render_tool_schema(tools)
    assert schema.startswith("<tools>\n<tool>{")
    assert '"name": "get_weather"' in schema
    assert schema.endswith("</tools>")

    tool_calls = [
        {
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"上海","date":"2026-06-03","unit":"celsius"}',
            }
        },
        {
            "function": {
                "name": "raw_xml_tool",
                "arguments": '<parameter name="city">上海</parameter>',
            }
        },
    ]
    rendered = render_tool_calls(tool_calls)
    assert rendered.startswith("<tool_call>\n<invoke name=\"get_weather\">")
    assert "<parameter name=\"city\">上海</parameter>" in rendered
    assert "<invoke name=\"raw_xml_tool\"><parameter name=\"city\">上海</parameter></invoke>" in rendered
    assert rendered.endswith("</tool_call>")


def test_render_tool_response_supports_string_list_and_plain_values():
    single = render_tool_response({"name": "demo", "content": "ok"})
    assert single == '<response name="demo">ok</response>'

    multi = render_tool_response(
        {
            "name": "demo",
            "content": [
                {"type": "text", "text": "alpha"},
                {"output": "beta"},
                {"x": 1},
                3,
            ],
        }
    )
    assert '<response name="demo">alpha\n</response>' in multi
    assert '<response name="demo">beta\n</response>' in multi
    assert '<response name="demo">{"x":1}\n</response>' in multi
    assert '<response name="demo">3\n</response>' in multi


def test_split_system_and_conversation_and_last_assistant_index():
    system, rest = split_system_and_conversation(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert system == {"role": "system", "content": "s"}
    assert rest == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert last_assistant_content_index(rest) == 1
    assert last_assistant_content_index([{"role": "user", "content": "u"}]) is None
    assert last_assistant_message(rest) == {"role": "assistant", "content": "a"}


def test_helper_functions_for_think_detection_and_message_slicing():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "final"},
    ]
    assert _assistant_has_existing_think("<think>\nfoo\n</think>\nbar") is True
    assert _assistant_has_existing_think("only </think>") is True
    assert _assistant_has_existing_think("plain") is False
    assert _messages_before_last_assistant(messages) == messages[:-1]
    no_assistant = [{"role": "user", "content": "u"}]
    assert _messages_before_last_assistant(no_assistant) == no_assistant
    assert _messages_with_normalized_final_assistant(no_assistant) == no_assistant
    assert last_assistant_message(no_assistant) is None

    normalized = _messages_with_normalized_final_assistant(messages)
    assert normalized[-1]["content"] == NO_THINKING_PREFIX + "final"

    keep_existing = _messages_with_normalized_final_assistant(
        [{"role": "assistant", "content": "<think>\nfoo\n</think>\nbar"}]
    )
    assert keep_existing[0]["content"] == "<think>\nfoo\n</think>\nbar"


def test_prepare_render_inputs_injects_system_message_when_overrides_provided():
    messages, tools = _prepare_render_inputs(
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        current_date="2026-06-03",
        current_location="Shanghai, China",
    )
    assert tools is None
    assert messages[0]["role"] == "system"
    assert messages[0]["current_date"] == "2026-06-03"
    assert messages[0]["current_location"] == "Shanghai, China"


def test_prepare_render_inputs_overrides_existing_system_message():
    messages, _ = _prepare_render_inputs(
        {
            "messages": [
                {"role": "system", "content": "base"},
                {"role": "assistant", "content": "a"},
            ]
        },
        current_date="2026-06-04",
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "base"
    assert messages[0]["current_date"] == "2026-06-04"


def test_build_template_segments_keeps_legacy_segment_shape_for_unit_checks():
    segments = build_template_segments(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
    )
    assert [segment.text for segment in segments if segment.trainable] == ["a2", "<|im_end|>"]

    tool_segments = build_template_segments(
        [
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "demo", "arguments": {"city": "上海"}}}],
            },
        ]
    )
    assert [segment.text for segment in tool_segments if segment.trainable] == [
        "\n",
        "<tool_call>\n<invoke name=\"demo\"><parameter name=\"city\">上海</parameter></invoke>\n</tool_call>",
        "<|im_end|>",
    ]


def test_build_template_segments_covers_tools_dates_and_tool_result_branch():
    segments = build_template_segments(
        [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "u"},
            {"role": "tool", "name": "demo", "content": "result"},
            {"role": "assistant", "content": "a\n"},
            {
                "role": "assistant",
                "content": "calls\n",
                "tool_calls": [{"function": {"name": "demo", "arguments": {"x": 1}}}],
            },
        ],
        tools=[{"function": {"name": "demo", "parameters": {"type": "object"}}}],
        current_date="2026-06-04",
        current_location="Beijing, China",
    )
    rendered = "".join(segment.text for segment in segments)
    assert "Current date: 2026-06-04" in rendered
    assert "Current location: Beijing, China" in rendered
    assert "<tools>" in rendered
    assert "<response name=\"demo\">" in rendered
    assert "calls\n<tool_call>" in rendered


def test_build_template_segments_supports_generation_prompt_and_errors():
    segments = build_template_segments(
        [{"role": "user", "content": "q"}],
        add_generation_prompt=True,
        enable_thinking=False,
        no_add_thinking=False,
    )
    assert "".join(segment.text for segment in segments).endswith(
        f"{IM_START_TOKEN}Assistant: {NO_THINKING_PREFIX}"
    )

    segments_no_think = build_template_segments(
        [{"role": "user", "content": "q"}],
        add_generation_prompt=True,
        no_add_thinking=True,
    )
    assert "".join(segment.text for segment in segments_no_think).endswith(f"{IM_START_TOKEN}Assistant: ")

    segments_open_think = build_template_segments(
        [{"role": "user", "content": "q"}],
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert "".join(segment.text for segment in segments_open_think).endswith(f"{IM_START_TOKEN}Assistant: <think>\n")

    with pytest.raises(ValueError, match="Unsupported role"):
        build_template_segments([{"role": "developer", "content": "x"}])


def test_render_chat_template_matches_my_sample_golden(chat_template, my_sample_record):
    rendered = render_chat_template(
        chat_template,
        normalize_record(my_sample_record)["messages"],
        tools=my_sample_record["tools"],
        no_add_thinking=False,
    )
    assert rendered == read_text_auto(MY_SAMPLE_OUTPUT)


def test_render_chat_template_matches_think_golden(chat_template, think_sample_record):
    rendered = render_chat_template(
        chat_template,
        normalize_record(think_sample_record)["messages"],
        tools=think_sample_record.get("tools"),
        no_add_thinking=True,
    )
    assert rendered == read_text_auto(THINK_SAMPLE_OUTPUT)


def test_render_chat_template_supports_generation_prompt_variants(chat_template):
    rendered_with_added_think = render_chat_template(
        chat_template,
        [{"role": "user", "content": "q"}],
        add_generation_prompt=True,
        enable_thinking=False,
        no_add_thinking=False,
    )
    assert rendered_with_added_think.endswith(f"{IM_START_TOKEN}Assistant: {NO_THINKING_PREFIX}")

    rendered_without_added_think = render_chat_template(
        chat_template,
        [{"role": "user", "content": "q"}],
        add_generation_prompt=True,
        enable_thinking=False,
        no_add_thinking=True,
    )
    assert rendered_without_added_think.endswith(f"{IM_START_TOKEN}Assistant: ")

    rendered_with_open_think = render_chat_template(
        chat_template,
        [{"role": "user", "content": "q"}],
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert rendered_with_open_think.endswith(f"{IM_START_TOKEN}Assistant: <think>\n")


def test_render_prefix_and_full_no_think_sample(chat_template, my_sample_record):
    prefix_text, full_text = _render_prefix_and_full(my_sample_record, template=chat_template)
    _, expected_no_think_text, _ = final_pattern_chunks()
    assert prefix_text.endswith(f"{IM_START_TOKEN}Assistant: {NO_THINKING_PREFIX}")
    assert full_text == expected_no_think_text.removesuffix(EOD_TOKEN)
    assert full_text.startswith(prefix_text)


def test_render_prefix_and_full_existing_think_sample(chat_template, think_sample_record):
    prefix_text, full_text = _render_prefix_and_full(think_sample_record, template=chat_template)
    assert prefix_text.endswith(f"{IM_START_TOKEN}Assistant: ")
    assert full_text == read_text_auto(THINK_SAMPLE_OUTPUT)
    assert full_text.startswith(prefix_text)


def test_render_prefix_and_full_handles_malformed_closing_think(chat_template):
    record = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "</think>\n答案"},
        ]
    }
    prefix_text, full_text = _render_prefix_and_full(record, template=chat_template)
    assert prefix_text.endswith(f"{IM_START_TOKEN}Assistant: ")
    assert f"{NO_THINKING_PREFIX}答案<|im_end|>\n" in full_text


def test_compute_trainable_span_and_char_mask(chat_template, my_sample_record):
    prefix_text, full_text = _render_prefix_and_full(my_sample_record, template=chat_template)
    start, end = _compute_trainable_span(full_text, prefix_text)
    assert start == len(prefix_text)
    assert end == len(full_text)
    empty_think_start = start - len(NO_THINKING_PREFIX)
    assert full_text[empty_think_start:start] == NO_THINKING_PREFIX

    char_mask = _char_mask_from_span(full_text, start, end)
    assert sum(char_mask[empty_think_start:start]) == 0
    assert sum(char_mask[:start]) == 0
    assert all(char_mask[pos] == 1 for pos in range(start, end))


def test_compute_trainable_span_rejects_non_matching_prefix():
    with pytest.raises(ValueError, match="rendered prefix"):
        _compute_trainable_span("full", "prefix")


def test_render_prefix_and_full_requires_assistant(chat_template):
    with pytest.raises(ValueError, match="assistant message"):
        _render_prefix_and_full({"messages": [{"role": "user", "content": "q"}]}, template=chat_template)


def test_eod_token_id_requires_single_token():
    class FakeTokenizer:
        def encode(self, text):
            return [1, 2]

    with pytest.raises(ValueError, match="exactly one token"):
        eod_token_id(FakeTokenizer())


def test_tokenize_with_char_spans_and_mask_projection(tokenizer: TRIE_TOKENIZER):
    text = "甲乙<|im_end|>"
    char_mask = [0, 1] + [1] * (len(text) - 2)
    token_ids, char_spans = _tokenize_with_char_spans(tokenizer, text)
    projected = _loss_mask_from_char_mask(char_mask, char_spans)

    assert tokenizer.decode(token_ids) == text
    assert len(token_ids) == len(char_spans) == len(projected)
    assert sum(projected) >= 1


def test_encode_segments_appends_eod_and_can_skip_it(tokenizer: TRIE_TOKENIZER):
    encoded = encode_segments(tokenizer, [Segment("abc", True), Segment("def", False)])
    assert tokenizer.decode(encoded.input_ids[:-1]) == "abcdef"
    assert encoded.loss_mask[-1] == 0

    encoded_with_trainable_eod = encode_segments(tokenizer, [Segment("abc", True)], eod_trainable=True)
    assert tokenizer.decode(encoded_with_trainable_eod.input_ids) == "abc<|endoftext|>"
    assert encoded_with_trainable_eod.loss_mask[-1] == 1

    no_eod = encode_segments(tokenizer, [Segment("abc", True)], append_eod=False)
    assert tokenizer.decode(no_eod.input_ids) == "abc"
    assert no_eod.loss_mask == [1] * len(no_eod.input_ids)


def test_eod_token_id_returns_single_endoftext_token(tokenizer: TRIE_TOKENIZER):
    assert eod_token_id(tokenizer) == tokenizer.token2idx[b"<|endoftext|>"]


def test_encode_separator_produces_masked_newline(tokenizer: TRIE_TOKENIZER):
    separator = _encode_separator(tokenizer)
    assert tokenizer.decode(separator.input_ids) == "\n"
    assert separator.loss_mask == [0] * len(separator.input_ids)


def test_build_document_from_record_trains_only_final_assistant_with_added_think(
    tokenizer: TRIE_TOKENIZER,
    chat_template,
    my_sample_record: dict,
):
    encoded = build_document_from_record(my_sample_record, tokenizer=tokenizer, template=chat_template)
    full_text = tokenizer.decode(encoded.input_ids[:-1])
    trainable_text = masked_text(tokenizer, encoded)
    _, expected_no_think_text, _ = final_pattern_chunks()

    assert full_text == expected_no_think_text.removesuffix(EOD_TOKEN)
    assert trainable_text.lstrip(" ") == (
        "上海今天多云，约 28°C，湿度 72%，东南风 3 级；空气质量为优，AQI 45。整体适合晚上跑步，建议避开闷热时段，控制强度并注意补水。"
        + "<|im_end|>\n<|endoftext|>"
    )
    assert NO_THINKING_PREFIX in full_text
    assert NO_THINKING_PREFIX not in trainable_text
    assert ASSISTANT_PREFIX not in trainable_text
    assert "<tool_call>" not in trainable_text
    assert encoded.loss_mask[-1] == 1


def test_build_document_from_record_trains_existing_think_sample(
    tokenizer: TRIE_TOKENIZER,
    chat_template,
    think_sample_record: dict,
):
    encoded = build_document_from_record(think_sample_record, tokenizer=tokenizer, template=chat_template)
    full_text = tokenizer.decode(encoded.input_ids[:-1])
    trainable_text = masked_text(tokenizer, encoded)

    assert full_text == read_text_auto(THINK_SAMPLE_OUTPUT)
    assert trainable_text.lstrip(" ") == "<think>\n刚才的结果是 180。\n180 / 4 = 45。\n</think>\n\n结果是 45。<|im_end|>\n<|endoftext|>"
    assert ASSISTANT_PREFIX not in trainable_text


def test_build_document_from_record_handles_malformed_closing_think_only(
    tokenizer: TRIE_TOKENIZER,
    chat_template,
):
    record = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "</think>\n答案"},
        ]
    }
    encoded = build_document_from_record(record, tokenizer=tokenizer, template=chat_template)
    assert masked_text(tokenizer, encoded).lstrip(" ") == f"{NO_THINKING_PREFIX}答案<|im_end|>\n<|endoftext|>"


def test_build_document_from_record_trains_final_tool_calls_when_content_empty(
    tokenizer: TRIE_TOKENIZER,
    chat_template,
):
    record = {
        "messages": [
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "上海"},
                        }
                    }
                ],
            },
        ]
    }
    encoded = build_document_from_record(record, tokenizer=tokenizer, template=chat_template)
    trainable_text = masked_text(tokenizer, encoded)
    assert trainable_text.lstrip(" ") == (
        "\n\n\n<tool_call>\n"
        + "<invoke name=\"get_weather\">\n"
        + "<parameter name=\"city\">上海</parameter>\n"
        + "</invoke>\n"
        + "</tool_call><|im_end|>\n<|endoftext|>"
    )
    assert NO_THINKING_PREFIX not in trainable_text


def test_build_document_from_record_uses_override_date_and_location(
    tokenizer: TRIE_TOKENIZER,
    chat_template,
):
    record = {
        "messages": [
            {"role": "assistant", "content": "答案"},
        ]
    }
    encoded = build_document_from_record(
        record,
        tokenizer=tokenizer,
        template=chat_template,
        current_date="2026-06-04",
        current_location="Beijing, China",
    )
    decoded = tokenizer.decode(encoded.input_ids[:-1])
    assert "Current date: 2026-06-04" in decoded
    assert "Current location: Beijing, China" in decoded


def test_pack_encoded_documents_pads_with_masked_eod_and_separator(tokenizer: TRIE_TOKENIZER):
    eod_id = eod_token_id(tokenizer)
    separator = _encode_separator(tokenizer)
    packed = list(
        pack_encoded_documents(
            [
                EncodedDocument(input_ids=[10, eod_id], loss_mask=[1, 1]),
                EncodedDocument(input_ids=[11, eod_id], loss_mask=[1, 1]),
            ],
            pack_length=6,
            pad_token_id=eod_id,
            separator=separator,
        )
    )
    assert len(packed) == 1
    assert tokenizer.decode(packed[0].input_ids) == f"\n".join(
        [tokenizer.decode([10, eod_id]), tokenizer.decode([11, eod_id])]
    ) + EOD_TOKEN
    assert packed[0].loss_mask[-1] == 0
    assert 0 in packed[0].loss_mask


def test_pack_encoded_documents_splits_long_stream_and_validates_lengths(tokenizer: TRIE_TOKENIZER):
    eod_id = eod_token_id(tokenizer)
    packed = list(
        pack_encoded_documents(
            [EncodedDocument(input_ids=[1, 2, 3, 4, eod_id], loss_mask=[1, 1, 1, 1, 1])],
            pack_length=3,
            pad_token_id=eod_id,
        )
    )
    assert [doc.input_ids for doc in packed] == [[1, 2, 3], [4, eod_id, eod_id]]
    assert [doc.loss_mask for doc in packed] == [[1, 1, 1], [1, 1, 0]]

    with pytest.raises(ValueError, match="positive integer"):
        list(pack_encoded_documents([], pack_length=0, pad_token_id=eod_id))

    with pytest.raises(ValueError, match="identical lengths"):
        list(
            pack_encoded_documents(
                [EncodedDocument(input_ids=[1, 2], loss_mask=[1])],
                pack_length=2,
                pad_token_id=eod_id,
            )
        )


def test_load_helpers_and_shuffle(tmp_path):
    template_path = tmp_path / "template.jinja"
    template_path.write_text("hello", encoding="utf-8")
    template = load_chat_template(str(template_path))
    assert hasattr(template, "render")

    input_path = tmp_path / "sample.jsonl"
    input_path.write_text("\n".join(["a", "", "b"]) + "\n", encoding="utf-8")
    assert load_non_empty_lines(str(input_path)) == ["a", "b"]

    shuffled = shuffled_epoch_lines(["a", "b"], 2, random.Random(0))
    assert sorted(shuffled) == ["a", "a", "b", "b"]
    not_shuffled = shuffled_epoch_lines(["a", "b"], 2, random.Random(0), shuffle=False)
    assert not_shuffled == ["a", "b", "a", "b"]


def test_load_jsonl_sources_supports_utf8_chinese_bom_and_parallel_reads(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_text(
        "\ufeff"
        + json.dumps({"messages": [{"role": "assistant", "content": "中文答案一"}]}, ensure_ascii=False)
        + "\n\n",
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "中文答案二"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sources = load_jsonl_sources([str(first_path), str(second_path)], num_workers=2)

    assert [source.line_number for source in sources] == [1, 1]
    assert [json.loads(source.text)["messages"][0]["content"] for source in sources] == [
        "中文答案一",
        "中文答案二",
    ]
    assert normalize_input_paths(str(first_path)) == [str(first_path)]
    assert normalize_input_paths([first_path, second_path]) == [str(first_path), str(second_path)]
    with pytest.raises(ValueError, match="At least one"):
        normalize_input_paths([])


def test_shuffled_epoch_sources_keeps_source_metadata():
    sources = [
        JsonlSourceLine(text="a", source_path="a.jsonl", line_number=1),
        JsonlSourceLine(text="b", source_path="b.jsonl", line_number=2),
    ]
    shuffled = shuffled_epoch_sources(sources, 2, random.Random(0))
    assert sorted((source.source_path, source.line_number) for source in shuffled) == [
        ("a.jsonl", 1),
        ("a.jsonl", 1),
        ("b.jsonl", 2),
        ("b.jsonl", 2),
    ]
    not_shuffled = shuffled_epoch_sources(sources, 2, random.Random(0), shuffle=False)
    assert [(source.source_path, source.line_number) for source in not_shuffled] == [
        ("a.jsonl", 1),
        ("b.jsonl", 2),
        ("a.jsonl", 1),
        ("b.jsonl", 2),
    ]


def test_compile_chat_template_and_env():
    env = _build_jinja_env()
    assert "tojson" in env.filters
    template = compile_chat_template("{{ value | tojson(ensure_ascii=False) }}")
    assert template.render(value={"x": "好"}) == '{"x": "好"}'


def test_path_helpers_and_default_output_prefix(tmp_path):
    prefix = str(tmp_path / "demo")
    assert index_file_path(prefix).endswith(".idx")
    assert data_file_path(prefix).endswith(".bin")
    assert mask_prefix_path(prefix).endswith(".mask")
    assert mask_file_path(prefix).endswith(".mask.bin")
    assert default_output_prefix(str(tmp_path / "file.jsonl")) == str((tmp_path / "file").resolve())
    assert default_output_prefix([str(tmp_path / "file.jsonl")]) == str((tmp_path / "file").resolve())
    with pytest.raises(ValueError, match="out-prefix"):
        default_output_prefix([str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")])


def test_write_documents_writes_token_and_mask_sidecar(tmp_path):
    output_prefix = str(tmp_path / "packed")
    stats = write_documents(
        output_prefix,
        [EncodedDocument(input_ids=[1, 2, 3], loss_mask=[0, 1, 0])],
        token_dtype=np.uint16,
        mask_dtype=np.uint8,
    )

    token_ds = MMapIndexedDataset(output_prefix)
    mask_ds = MMapIndexedDataset(output_prefix + ".mask")
    assert token_ds[0].astype(int).tolist() == [1, 2, 3]
    assert mask_ds[0].astype(int).tolist() == [0, 1, 0]
    assert stats["documents"] == 1
    assert stats["tokens"] == 3
    assert stats["trainable_tokens"] == 1
    assert stats["mask_path"].endswith(".mask.bin")


def test_write_documents_rejects_mask_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="identical lengths"):
        write_documents(
            str(tmp_path / "bad"),
            [EncodedDocument(input_ids=[1, 2], loss_mask=[1])],
        )


def test_build_binidx_dataset_writes_token_and_mask_sidecar_for_my_sample(tmp_path):
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(MY_SAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    output_prefix = str(tmp_path / "packed")

    stats = build_binidx_dataset(
        str(input_path),
        output_prefix=output_prefix,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=1,
        seed=7,
    )

    token_ds = MMapIndexedDataset(output_prefix)
    mask_ds = MMapIndexedDataset(output_prefix + ".mask")
    tokens = token_ds[0].astype(int).tolist()
    mask = mask_ds[0].astype(int).tolist()

    assert stats["documents"] == 1
    assert stats["source_documents"] == 1
    assert stats["source_lines"] == 1
    assert stats["epochs"] == 1
    assert stats["pack_length"] is None
    assert len(tokens) == len(mask)
    assert sum(mask) > 0
    _, expected_no_think_text, _ = final_pattern_chunks()
    tokenizer_obj = TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)
    assert tokenizer_decode(tokenizer_obj, tokens) == expected_no_think_text + EOD_TOKEN
    assert tokens[-1] == 65532
    assert mask[-1] == 1


def tokenizer_decode(tokenizer: TRIE_TOKENIZER, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids)


def test_build_binidx_dataset_matches_final_pattern_when_packed(tmp_path):
    think_record, no_think_record = final_pattern_records()
    input_path = tmp_path / "packed.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(think_record, ensure_ascii=False),
                json.dumps(no_think_record, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_prefix = str(tmp_path / "packed_fixed")
    tokenizer_obj = TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)
    expected_text = read_text_auto(FINAL_PATTERN_PATH)
    pack_length = len(tokenizer_obj.encode(expected_text))

    stats = build_binidx_dataset(
        str(input_path),
        output_prefix=output_prefix,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=1,
        seed=7,
        pack_length=pack_length,
    )

    token_ds = MMapIndexedDataset(output_prefix)
    mask_ds = MMapIndexedDataset(output_prefix + ".mask")
    tokens = token_ds[0].astype(int).tolist()
    mask = mask_ds[0].astype(int).tolist()
    decoded = tokenizer_obj.decode(tokens)

    assert stats["documents"] == 1
    assert stats["source_documents"] == 2
    assert stats["pack_length"] == pack_length
    assert decoded == expected_text
    assert len(tokens) == len(mask) == pack_length

    expected_first_text, _, _ = final_pattern_chunks()
    separator_ids = tokenizer_obj.encode("\n")
    separator_start = len(tokenizer_obj.encode(expected_first_text))
    assert tokens[separator_start:separator_start + len(separator_ids)] == separator_ids
    assert mask[separator_start:separator_start + len(separator_ids)] == [0] * len(separator_ids)


def test_build_binidx_dataset_padding_keeps_tail_eod_mask_zero(tmp_path):
    tokenizer_obj = TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)
    input_path = tmp_path / "single.jsonl"
    input_path.write_text(THINK_SAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    output_prefix = str(tmp_path / "single_out")
    document = build_document_from_record(
        json.loads(THINK_SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0]),
        tokenizer=tokenizer_obj,
        template=load_chat_template(str(TEMPLATE_PATH)),
    )
    pack_length = len(document.input_ids) + 3

    build_binidx_dataset(
        str(input_path),
        output_prefix=output_prefix,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=1,
        seed=7,
        pack_length=pack_length,
    )
    mask_ds = MMapIndexedDataset(output_prefix + ".mask")
    mask = mask_ds[0].astype(int).tolist()
    assert mask[-3:] == [0, 0, 0]


def test_build_binidx_dataset_shuffles_epochs_deterministically(tmp_path):
    input_path = tmp_path / "sample.jsonl"
    records = [
        {"messages": [{"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "assistant", "content": "b"}]},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    prefix_one = str(tmp_path / "one")
    prefix_two = str(tmp_path / "two")

    build_binidx_dataset(
        str(input_path),
        output_prefix=prefix_one,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=3,
        seed=11,
    )
    build_binidx_dataset(
        str(input_path),
        output_prefix=prefix_two,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=3,
        seed=11,
    )

    assert Path(data_file_path(prefix_one)).read_bytes() == Path(data_file_path(prefix_two)).read_bytes()
    assert Path(mask_file_path(prefix_one)).read_bytes() == Path(mask_file_path(prefix_two)).read_bytes()


def test_build_binidx_dataset_accepts_multiple_utf8_jsonl_files_with_workers(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_record = {"messages": [{"role": "assistant", "content": "中文答案一"}]}
    second_record = {"messages": [{"role": "assistant", "content": "中文答案二"}]}
    first_path.write_text(json.dumps(first_record, ensure_ascii=False) + "\n", encoding="utf-8")
    second_path.write_text(json.dumps(second_record, ensure_ascii=False) + "\n", encoding="utf-8")
    serial_prefix = str(tmp_path / "serial")
    parallel_prefix = str(tmp_path / "parallel")

    serial_stats = build_binidx_dataset(
        [str(first_path), str(second_path)],
        output_prefix=serial_prefix,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=2,
        seed=123,
        num_workers=1,
    )
    parallel_stats = build_binidx_dataset(
        [str(first_path), str(second_path)],
        output_prefix=parallel_prefix,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=2,
        seed=123,
        num_workers=2,
    )

    assert serial_stats["source_files"] == parallel_stats["source_files"] == 2
    assert serial_stats["source_lines"] == parallel_stats["source_lines"] == 2
    assert serial_stats["source_documents"] == parallel_stats["source_documents"] == 4
    assert parallel_stats["num_workers"] == 2
    assert Path(data_file_path(serial_prefix)).read_bytes() == Path(data_file_path(parallel_prefix)).read_bytes()
    assert Path(mask_file_path(serial_prefix)).read_bytes() == Path(mask_file_path(parallel_prefix)).read_bytes()

    tokenizer_obj = TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)
    token_ds = MMapIndexedDataset(parallel_prefix)
    mask_ds = MMapIndexedDataset(parallel_prefix + ".mask")
    decoded_documents = [tokenizer_obj.decode(token_ds[index].astype(int).tolist()) for index in range(len(token_ds))]
    trainable_documents = [
        tokenizer_obj.decode(
            [
                token_id
                for token_id, keep in zip(
                    token_ds[index].astype(int).tolist(),
                    mask_ds[index].astype(int).tolist(),
                )
                if keep == 1
            ]
        )
        for index in range(len(token_ds))
    ]
    assert any("中文答案一" in text for text in decoded_documents)
    assert any("中文答案二" in text for text in decoded_documents)
    assert any("中文答案一" in text for text in trainable_documents)
    assert any("中文答案二" in text for text in trainable_documents)


def test_build_binidx_dataset_can_disable_shuffle_for_ordered_epochs(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "first-answer"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "second-answer"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_prefix = str(tmp_path / "ordered")

    stats = build_binidx_dataset(
        [str(first_path), str(second_path)],
        output_prefix=output_prefix,
        vocab_path=str(VOCAB_PATH),
        template_path=str(TEMPLATE_PATH),
        n_epoch=2,
        seed=999,
        num_workers=2,
        shuffle=False,
    )

    tokenizer_obj = TRIE_TOKENIZER(str(VOCAB_PATH), strict_length=True)
    token_ds = MMapIndexedDataset(output_prefix)
    decoded_documents = [tokenizer_obj.decode(token_ds[index].astype(int).tolist()) for index in range(len(token_ds))]
    assert stats["shuffle"] is False
    assert ["first-answer" in text for text in decoded_documents] == [True, False, True, False]
    assert ["second-answer" in text for text in decoded_documents] == [False, True, False, True]


def test_build_documents_from_sources_reports_json_errors(tokenizer: TRIE_TOKENIZER, chat_template):
    sources = [JsonlSourceLine(text="{bad json", source_path="bad.jsonl", line_number=3)]
    with pytest.raises(ValueError, match=r"bad\.jsonl:3"):
        list(
            build_documents_from_sources(
                sources,
                tokenizer=tokenizer,
                template=chat_template,
                num_workers=2,
            )
        )


def test_build_binidx_dataset_rejects_invalid_worker_count(tmp_path):
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "a"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_workers"):
        build_binidx_dataset(
            str(input_path),
            output_prefix=str(tmp_path / "out"),
            vocab_path=str(VOCAB_PATH),
            template_path=str(TEMPLATE_PATH),
            n_epoch=1,
            seed=1,
            num_workers=0,
        )


def test_cli_main_builds_dataset_and_accepts_flags(tmp_path):
    input_path = tmp_path / "cli.jsonl"
    input_path.write_text(THINK_SAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    output_prefix = tmp_path / "cli_out"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        make_sft_binidx_main(
            [
                str(input_path),
                "--out-prefix",
                str(output_prefix),
                "--vocab",
                str(VOCAB_PATH),
                "--chat-template",
                str(TEMPLATE_PATH),
                "--current-date",
                "2026-06-03",
                "--current-location",
                "Shanghai, China",
                "--pack-length",
                "128",
                "--num-workers",
                "2",
                "--no-shuffle",
                "--add-generation-prompt",
                "--enable-thinking",
            ]
        )

    output = stdout.getvalue()
    assert "Built SFT binidx dataset" in output
    assert (tmp_path / "cli_out.bin").exists()
    assert (tmp_path / "cli_out.idx").exists()
    assert Path(mask_file_path(str(output_prefix))).exists()
    assert (tmp_path / "cli_out.mask.idx").exists()


def test_arg_parser_defaults_and_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(["sample.jsonl"])
    assert args.input_jsonl == ["sample.jsonl"]
    assert args.vocab == "rwkv_vocab_v20260603.txt"
    assert args.chat_template == "data/SFT/sample/chat_template.jinja"
    assert args.n_epoch == 1
    assert args.seed == 1234
    assert args.num_workers == 1
    assert args.shuffle is True

    overridden = parser.parse_args(
        [
            "sample.jsonl",
            "sample2.jsonl",
            "--out-prefix",
            "out",
            "--current-date",
            "2026-06-03",
            "--current-location",
            "Shanghai",
            "--pack-length",
            "64",
            "--add-generation-prompt",
            "--enable-thinking",
            "--n-epoch",
            "2",
            "--seed",
            "9",
            "--num-workers",
            "3",
            "--no-shuffle",
        ]
    )
    assert overridden.input_jsonl == ["sample.jsonl", "sample2.jsonl"]
    assert overridden.out_prefix == "out"
    assert overridden.current_date == "2026-06-03"
    assert overridden.current_location == "Shanghai"
    assert overridden.pack_length == 64
    assert overridden.add_generation_prompt is True
    assert overridden.enable_thinking is True
    assert overridden.n_epoch == 2
    assert overridden.seed == 9
    assert overridden.num_workers == 3
    assert overridden.shuffle is False


def test_tools_jsonl_smoke_still_only_trains_last_assistant_if_available(
    tokenizer: TRIE_TOKENIZER,
    chat_template,
):
    first_line = TOOLS_JSONL.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    encoded = build_document_from_record(record, tokenizer=tokenizer, template=chat_template)
    trainable_text = masked_text(tokenizer, encoded)

    last_assistant = next(
        visible_text(message["content"])
        for message in reversed(record["messages"])
        if message["role"] == "assistant"
    )
    assert trainable_text.endswith("<|im_end|>\n<|endoftext|>")
    if last_assistant.strip():
        assert last_assistant.strip() in trainable_text
    else:
        assert "<tool_call>" in trainable_text

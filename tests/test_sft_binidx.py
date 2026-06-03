import io
import json
import random
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
    EOD_TOKEN,
    EncodedDocument,
    Segment,
    build_binidx_dataset,
    build_document_from_record,
    build_system_text,
    build_template_segments,
    data_file_path,
    default_output_prefix,
    encode_segments,
    index_file_path,
    last_assistant_content_index,
    load_chat_template,
    load_non_empty_lines,
    mask_file_path,
    mask_prefix_path,
    parse_tool_arguments,
    render_chat_template,
    render_tool_calls,
    render_tool_response,
    render_tool_schema,
    shuffled_epoch_lines,
    split_system_and_conversation,
    visible_text,
    write_documents,
)


VOCAB_PATH = ROOT / "rwkv_vocab_v20260603.txt"
TEMPLATE_PATH = ROOT / "chat_template.jinja"
TOOLS_JSONL = ROOT / "data" / "SFT" / "tools.jsonl"
ORIGIN_EXAMPLE = ROOT / "data" / "SFT" / "stf_origin_example.jsonl"
TEMPLATE_EXAMPLE = ROOT / "data" / "SFT" / "stf_template_example.txt"


@pytest.fixture(scope="module")
def tokenizer():
    return TRIE_TOKENIZER(str(VOCAB_PATH))


@pytest.fixture(scope="module")
def chat_template():
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def example_record():
    return json.loads(ORIGIN_EXAMPLE.read_text(encoding="utf-8"))


def masked_text(tokenizer: TRIE_TOKENIZER, encoded: EncodedDocument) -> str:
    kept = [
        token_id
        for token_id, mask in zip(encoded.input_ids[:-1], encoded.loss_mask[:-1])
        if mask == 1
    ]
    return tokenizer.decode(kept)


def test_parse_vocab_line_accepts_new_vocab_spacing_mismatches():
    idx, token_bytes, declared_length = parse_vocab_line("19231 ' ' 3")
    assert idx == 19231
    assert token_bytes == b" "
    assert declared_length == 3


def test_parse_vocab_line_strict_length_raises():
    with pytest.raises(ValueError, match="Token length mismatch"):
        parse_vocab_line("1 'a' 2", strict_length=True)


def test_parse_vocab_line_rejects_non_byte_tokens():
    with pytest.raises(TypeError, match="Unsupported token value type"):
        parse_vocab_line("7 123 3")


def test_new_vocab_loads_special_tokens_and_warns_for_len_mismatches():
    with pytest.warns(RuntimeWarning, match="declared byte lengths"):
        loaded = TRIE_TOKENIZER(str(VOCAB_PATH))
    assert loaded.token2idx[b"<|im_start|>"] == 65530
    assert loaded.token2idx[b"<|im_end|>"] == 65531
    assert loaded.token2idx[b"<|endoftext|>"] == 65532


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
    assert '"name":"get_weather"' in schema
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


def test_build_template_segments_matches_origin_example(example_record):
    segments = build_template_segments(example_record["messages"], tools=example_record["tools"])
    rendered = "".join(segment.text for segment in segments)
    expected = TEMPLATE_EXAMPLE.read_text(encoding="utf-8")
    assert rendered == expected

    trainable_texts = [segment.text for segment in segments if segment.trainable]
    assert trainable_texts == [
        "上海今天多云，约 28°C，湿度 72%，东南风 3 级；空气质量为优，AQI 45。整体适合晚上跑步，建议避开闷热时段，控制强度并注意补水。"
    ]


def test_build_template_segments_marks_only_last_assistant_content():
    segments = build_template_segments(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
    )
    assert [segment.text for segment in segments if segment.trainable] == ["a2"]


def test_build_template_segments_supports_generation_prompt_and_tool_runs():
    segments = build_template_segments(
        [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "demo", "arguments": {"city": "上海"}}}],
            },
            {"role": "tool", "name": "demo", "content": [{"type": "text", "text": "tool output"}]},
        ],
        tools=[{"function": {"name": "demo", "parameters": {"type": "object"}}}],
        add_generation_prompt=True,
        enable_thinking=True,
    )
    rendered = "".join(segment.text for segment in segments)
    assert rendered.endswith("<|im_start|>Assistant: <think>")
    assert "<|im_start|>Tool: " in rendered
    assert "<response name=\"demo\">tool output\n</response><|im_end|>\n" in rendered


def test_build_template_segments_without_thinking_prompt_and_unsupported_role():
    rendered = "".join(
        segment.text
        for segment in build_template_segments(
            [{"role": "user", "content": "q"}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    assert rendered.endswith("<|im_start|>Assistant: <think>\n</think>")

    with pytest.raises(ValueError, match="Unsupported role"):
        build_template_segments([{"role": "developer", "content": "x"}])


def test_render_chat_template_ignores_template_text_and_matches_example(example_record, chat_template):
    rendered = render_chat_template(chat_template, example_record["messages"], tools=example_record["tools"])
    assert rendered == TEMPLATE_EXAMPLE.read_text(encoding="utf-8")


def test_only_last_assistant_content_is_trainable(tokenizer: TRIE_TOKENIZER, chat_template: str):
    record = {
        "messages": [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "tool", "content": "{\"ok\":true}", "name": "demo_tool"},
            {"role": "assistant", "content": "最后答案"},
        ]
    }

    rendered = render_chat_template(chat_template, record["messages"])
    encoded = build_document_from_record(record, tokenizer=tokenizer, template=chat_template)
    assert tokenizer.decode(encoded.input_ids[:-1]) == rendered

    trainable_text = masked_text(tokenizer, encoded)
    assert trainable_text == "最后答案"
    assert "<|im_start|>Assistant: " not in trainable_text
    assert "<tool_call>" not in trainable_text
    assert encoded.loss_mask[-1] == 0


def test_origin_example_only_trains_last_assistant_content(
    tokenizer: TRIE_TOKENIZER,
    chat_template: str,
    example_record: dict,
):
    encoded = build_document_from_record(example_record, tokenizer=tokenizer, template=chat_template)
    trainable_text = masked_text(tokenizer, encoded)

    assert trainable_text == example_record["messages"][-1]["content"]
    assert "帮我查一下上海今天的天气" not in trainable_text
    assert "get_weather" not in trainable_text
    assert EOD_TOKEN not in trainable_text


def test_encode_segments_appends_eod_and_can_skip_it(tokenizer: TRIE_TOKENIZER):
    encoded = encode_segments(tokenizer, [Segment("abc", True), Segment("def", False)])
    assert tokenizer.decode(encoded.input_ids[:-1]) == "abcdef"
    assert encoded.loss_mask[-1] == 0

    no_eod = encode_segments(tokenizer, [Segment("abc", True)], append_eod=False)
    assert tokenizer.decode(no_eod.input_ids) == "abc"
    assert no_eod.loss_mask == [1] * len(no_eod.input_ids)


def test_build_document_from_record_ignores_template_text(tokenizer: TRIE_TOKENIZER):
    record = {"messages": [{"role": "assistant", "content": "answer"}]}
    encoded = build_document_from_record(record, tokenizer=tokenizer, template="manually different")
    assert masked_text(tokenizer, encoded) == "answer"


def test_load_helpers_and_shuffle(tmp_path):
    template_path = tmp_path / "template.jinja"
    template_path.write_text("hello", encoding="utf-8")
    assert load_chat_template(str(template_path)) == "hello"

    input_path = tmp_path / "sample.jsonl"
    input_path.write_text("\n".join(["a", "", "b"]) + "\n", encoding="utf-8")
    assert load_non_empty_lines(str(input_path)) == ["a", "b"]

    shuffled = shuffled_epoch_lines(["a", "b"], 2, random.Random(0))
    assert sorted(shuffled) == ["a", "a", "b", "b"]


def test_path_helpers_and_default_output_prefix(tmp_path):
    prefix = str(tmp_path / "demo")
    assert index_file_path(prefix).endswith(".idx")
    assert data_file_path(prefix).endswith(".bin")
    assert mask_prefix_path(prefix).endswith(".mask")
    assert mask_file_path(prefix).endswith(".mask.bin")
    assert default_output_prefix(str(tmp_path / "file.jsonl")) == str((tmp_path / "file").resolve())


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


def test_build_binidx_dataset_writes_token_and_mask_sidecar(tmp_path):
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "答案"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
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
    assert stats["source_lines"] == 1
    assert stats["epochs"] == 1
    assert len(tokens) == len(mask)
    assert sum(mask) > 0
    assert tokens[-1] == 65532
    assert mask[-1] == 0


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


def test_cli_main_builds_dataset_and_accepts_flags(tmp_path):
    input_path = tmp_path / "cli.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
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
    assert args.vocab == "rwkv_vocab_v20260603.txt"
    assert args.chat_template == "chat_template.jinja"
    assert args.n_epoch == 1
    assert args.seed == 1234

    overridden = parser.parse_args(
        [
            "sample.jsonl",
            "--out-prefix",
            "out",
            "--current-date",
            "2026-06-03",
            "--current-location",
            "Shanghai",
            "--add-generation-prompt",
            "--enable-thinking",
            "--n-epoch",
            "2",
            "--seed",
            "9",
        ]
    )
    assert overridden.out_prefix == "out"
    assert overridden.current_date == "2026-06-03"
    assert overridden.current_location == "Shanghai"
    assert overridden.add_generation_prompt is True
    assert overridden.enable_thinking is True
    assert overridden.n_epoch == 2
    assert overridden.seed == 9


def test_tools_jsonl_smoke_still_only_trains_last_assistant_if_available(
    tokenizer: TRIE_TOKENIZER,
    chat_template: str,
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
    assert trainable_text == last_assistant

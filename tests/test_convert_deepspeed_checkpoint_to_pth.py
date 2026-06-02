import importlib.util
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_deepspeed_checkpoint_to_pth.py"
SPEC = importlib.util.spec_from_file_location("convert_deepspeed_checkpoint_to_pth", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
convert_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert_script)


def test_normalize_param_name_strips_forward_module_prefix():
    assert convert_script.normalize_param_name("_forward_module.blocks.0.att.key.weight") == "blocks.0.att.key.weight"
    assert convert_script.normalize_param_name("blocks.0.att.key.weight") == "blocks.0.att.key.weight"


def test_cast_tensor_dtype_changes_floating_tensors_only():
    float_tensor = torch.randn(2, 3, dtype=torch.float32)
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

    cast_float = convert_script.cast_tensor_dtype(float_tensor, torch.bfloat16)
    cast_int = convert_script.cast_tensor_dtype(int_tensor, torch.bfloat16)

    assert cast_float.dtype == torch.bfloat16
    assert cast_int.dtype == torch.int64


def test_build_and_parse_summary_lines_round_trip():
    state_dict = OrderedDict(
        [
            ("emb.weight", torch.zeros(4, 8, dtype=torch.bfloat16)),
            ("ln_out.bias", torch.zeros(8, dtype=torch.bfloat16)),
        ]
    )
    lines = convert_script.build_summary_lines(state_dict)
    params, total = convert_script.parse_summary_lines(lines)

    assert params == [
        ("emb.weight", (4, 8), "torch.bfloat16"),
        ("ln_out.bias", (8,), "torch.bfloat16"),
    ]
    assert total == 40


def test_materialize_state_dict_casts_and_normalizes_names():
    raw_state_dict = OrderedDict(
        [
            ("_forward_module.emb.weight", torch.ones(2, 2, dtype=torch.float32)),
            ("ln_out.bias", torch.ones(2, dtype=torch.float32)),
        ]
    )
    converted = convert_script.materialize_state_dict(raw_state_dict, torch.bfloat16)

    assert list(converted.keys()) == ["emb.weight", "ln_out.bias"]
    assert all(t.dtype == torch.bfloat16 for t in converted.values())


def test_verify_summary_accepts_matching_reference(tmp_path: Path):
    state_dict = OrderedDict(
        [
            ("emb.weight", torch.zeros(2, 2, dtype=torch.bfloat16)),
            ("ln_out.bias", torch.zeros(2, dtype=torch.bfloat16)),
        ]
    )
    reference = tmp_path / "reference.txt"
    reference.write_text("\n".join(convert_script.build_summary_lines(state_dict)) + "\n", encoding="utf-8")

    convert_script.verify_summary(reference, state_dict)


def test_verify_summary_rejects_mismatch(tmp_path: Path):
    state_dict = OrderedDict(
        [
            ("emb.weight", torch.zeros(2, 2, dtype=torch.bfloat16)),
        ]
    )
    reference = tmp_path / "reference.txt"
    reference.write_text(
        "Parameter: emb.weight, Shape: torch.Size([2, 3]), Dtype: torch.bfloat16\nTotal parameters: 6\n",
        encoding="utf-8",
    )

    try:
        convert_script.verify_summary(reference, state_dict)
    except ValueError as exc:
        assert "Summary mismatch" in str(exc) or "Total parameter mismatch" in str(exc)
    else:
        raise AssertionError("Expected verify_summary to raise on mismatched reference")


def test_looks_like_deepspeed_checkpoint_dir_by_contents_without_pth_suffix(tmp_path: Path):
    checkpoint_dir = tmp_path / "global_step200"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "zero_to_fp32.py").write_text("# stub\n", encoding="utf-8")

    assert convert_script.looks_like_deepspeed_checkpoint_dir(checkpoint_dir) is True

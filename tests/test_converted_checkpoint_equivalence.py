import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "test_converted_checkpoint_equivalence.py"
SPEC = importlib.util.spec_from_file_location("test_converted_checkpoint_equivalence_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
equiv_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(equiv_script)


def test_infer_demo_dims_from_state_dict():
    state_dict = OrderedDict(
        [
            ("emb.weight", torch.zeros(65536, 4096, dtype=torch.bfloat16)),
            ("blocks.0.ln1.weight", torch.zeros(4096, dtype=torch.bfloat16)),
            ("blocks.1.ln1.weight", torch.zeros(4096, dtype=torch.bfloat16)),
            ("blocks.0.att.w1", torch.zeros(4096, 192, dtype=torch.bfloat16)),
            ("blocks.0.att.a1", torch.zeros(4096, 160, dtype=torch.bfloat16)),
            ("blocks.0.att.v1", torch.zeros(4096, 96, dtype=torch.bfloat16)),
            ("blocks.0.att.g1", torch.zeros(4096, 384, dtype=torch.bfloat16)),
        ]
    )

    dims = equiv_script.infer_demo_dims_from_state_dict(state_dict)

    assert dims == {
        "n_layer": 2,
        "n_embd": 4096,
        "vocab_size": 65536,
        "d_decay_lora": 192,
        "d_aaa_lora": 160,
        "d_mv_lora": 96,
        "d_gate_lora": 384,
    }


def test_compare_state_dicts_accepts_exact_match():
    lhs = OrderedDict(
        [
            ("emb.weight", torch.ones(2, 2, dtype=torch.bfloat16)),
            ("ln_out.bias", torch.zeros(2, dtype=torch.int64)),
        ]
    )
    rhs = OrderedDict(
        [
            ("emb.weight", torch.ones(2, 2, dtype=torch.bfloat16)),
            ("ln_out.bias", torch.zeros(2, dtype=torch.int64)),
        ]
    )

    summary = equiv_script.compare_state_dicts(lhs, rhs, max_abs_tol=0.0, max_rel_tol=0.0)

    assert summary["checked_tensors"] == 2
    assert summary["max_abs_diff"] == 0.0
    assert summary["max_rel_diff"] == 0.0


def test_compare_state_dicts_rejects_tensor_mismatch():
    lhs = OrderedDict([("emb.weight", torch.tensor([[1.0]], dtype=torch.float32))])
    rhs = OrderedDict([("emb.weight", torch.tensor([[2.0]], dtype=torch.float32))])

    with pytest.raises(ValueError, match="Tensor mismatch"):
        equiv_script.compare_state_dicts(lhs, rhs, max_abs_tol=0.0, max_rel_tol=0.0)


def test_compare_state_dicts_rejects_key_mismatch():
    lhs = OrderedDict([("emb.weight", torch.tensor([[1.0]], dtype=torch.float32))])
    rhs = OrderedDict([("head.weight", torch.tensor([[1.0]], dtype=torch.float32))])

    with pytest.raises(ValueError, match="State dict keys differ"):
        equiv_script.compare_state_dicts(lhs, rhs, max_abs_tol=0.0, max_rel_tol=0.0)


def test_load_demo_module_uses_repo_runtime_module():
    module = equiv_script.load_demo_module()

    assert module.__name__ == "rwkv_v7_demo_runtime"
    assert Path(module.__file__).name == "rwkv_v7_demo_runtime.py"
    assert hasattr(module, "configure_runtime")
    assert hasattr(module, "RWKV")
    assert hasattr(module, "RWKV_TOKENIZER")

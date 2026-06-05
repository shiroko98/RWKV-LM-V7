import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_state_dict(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    normalized = OrderedDict()
    for key, value in state.items():
        clean_key = key.removeprefix("_forward_module.")
        normalized[clean_key] = value
    return normalized


def _infer_rwkv7_dims(state_dict) -> dict[str, int]:
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    return {
        "n_layer": len(layer_ids),
        "n_embd": int(state_dict["emb.weight"].shape[1]),
        "vocab_size": int(state_dict["emb.weight"].shape[0]),
        "dim_ffn": int(state_dict["blocks.0.ffn.key.weight"].shape[0]),
        "head_size": int(state_dict["blocks.0.att.r_k"].shape[1]),
        "d_decay_lora": int(state_dict["blocks.0.att.w1"].shape[1]),
        "d_aaa_lora": int(state_dict["blocks.0.att.a1"].shape[1]),
        "d_mv_lora": int(state_dict["blocks.0.att.v1"].shape[1]),
        "d_gate_lora": int(state_dict["blocks.0.att.g1"].shape[1]),
    }


def _require_cuda_smoke(env_name: str) -> Path:
    if os.environ.get(env_name) != "1":
        pytest.skip(f"set {env_name}=1 to run this CUDA SFT smoke test")
    model_path = os.environ.get("RWKV_SFT_SMOKE_MODEL", "")
    if not model_path:
        pytest.skip("set RWKV_SFT_SMOKE_MODEL to a RWKV7 .pth checkpoint")
    path = Path(model_path).expanduser()
    if not path.is_file():
        pytest.skip(f"RWKV_SFT_SMOKE_MODEL does not exist: {path}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return path


def _build_tiny_sft_binidx(tmp_path: Path, pad_length: int) -> Path:
    from src.sft_binidx import build_binidx_dataset

    prefix = tmp_path / "tiny_sft"
    stats = build_binidx_dataset(
        [str(ROOT / "data" / "SFT" / "sample" / "think.jsonl")],
        output_prefix=str(prefix),
        vocab_path=str(ROOT / "rwkv_vocab_v20260603.txt"),
        template_path=str(ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"),
        pad_length=pad_length,
        num_workers=1,
        shuffle=False,
    )
    assert stats["documents"] == 1
    assert stats["trainable_tokens"] > 0
    return prefix


def _base_sft_args(prefix: Path, dims: dict[str, int], ctx_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=dims["vocab_size"],
        data_file=str(prefix),
        data_type="sft_binidx",
        sft_mask_file="",
        sft_pad_token_id=65532,
        epoch_steps=1,
        epoch_count=1,
        real_bsz=1,
        train_stage=0,
        ctx_len=ctx_len,
        magic_prime=0,
        epoch_begin=0,
        resume_epoch=0,
        resume_step_offset=0,
        micro_bsz=1,
    )


def test_infer_rwkv7_dims_from_sft_smoke_state_dict():
    state = OrderedDict(
        [
            ("emb.weight", torch.zeros(65536, 1024, dtype=torch.bfloat16)),
            ("blocks.0.ffn.key.weight", torch.zeros(4096, 1024, dtype=torch.bfloat16)),
            ("blocks.0.att.r_k", torch.zeros(16, 64, dtype=torch.bfloat16)),
            ("blocks.0.att.w1", torch.zeros(1024, 64, dtype=torch.bfloat16)),
            ("blocks.0.att.a1", torch.zeros(1024, 64, dtype=torch.bfloat16)),
            ("blocks.0.att.v1", torch.zeros(1024, 32, dtype=torch.bfloat16)),
            ("blocks.0.att.g1", torch.zeros(1024, 128, dtype=torch.bfloat16)),
            ("blocks.1.ln1.weight", torch.zeros(1024, dtype=torch.bfloat16)),
        ]
    )

    assert _infer_rwkv7_dims(state) == {
        "n_layer": 2,
        "n_embd": 1024,
        "vocab_size": 65536,
        "dim_ffn": 4096,
        "head_size": 64,
        "d_decay_lora": 64,
        "d_aaa_lora": 64,
        "d_mv_lora": 32,
        "d_gate_lora": 128,
    }


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_sft_checkpoint_forward_backward(tmp_path, monkeypatch):
    model_path = _require_cuda_smoke("RWKV_RUN_CUDA_SFT_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257"))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)

    monkeypatch.setenv("RWKV_MY_TESTING", os.environ.get("RWKV_SFT_SMOKE_MY_TESTING", "x070"))
    monkeypatch.setenv("RWKV_KERNEL", os.environ.get("RWKV_SFT_SMOKE_KERNEL", ""))
    monkeypatch.setenv("RWKV_JIT_ON", "1")
    monkeypatch.setenv("RWKV_CTXLEN", str(ctx_len))
    monkeypatch.setenv("RWKV_HEAD_SIZE", str(dims["head_size"]))
    monkeypatch.setenv("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    monkeypatch.setenv("RWKV_FLOAT_MODE", "bf16")

    from src.dataset import MyDataset
    from src.model import RWKV

    model_args = SimpleNamespace(
        **dims,
        dim_att=dims["n_embd"],
        my_testing=os.environ["RWKV_MY_TESTING"],
        grad_cp=0,
        ctx_len=ctx_len,
    )
    model = RWKV(model_args).to(device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    model.train()

    dataset = MyDataset(_base_sft_args(prefix, dims, ctx_len))
    batch = tuple(t.unsqueeze(0).cuda(non_blocking=True) for t in dataset[0])
    assert batch[2].sum().item() > 0

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss.detach()).item()
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_cuda_one_step(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257"))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    proj_dir = tmp_path / "out"
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_2")

    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--load_model",
        str(model_path),
        "--wandb",
        "",
        "--proj_dir",
        str(proj_dir),
        "--data_file",
        str(prefix),
        "--data_type",
        "sft_binidx",
        "--ctx_len",
        str(ctx_len),
        "--epoch_steps",
        "1",
        "--epoch_count",
        "1",
        "--micro_bsz",
        "1",
        "--my_exit_tokens",
        "0",
        "--vocab_size",
        str(dims["vocab_size"]),
        "--n_layer",
        str(dims["n_layer"]),
        "--n_embd",
        str(dims["n_embd"]),
        "--dim_ffn",
        str(dims["dim_ffn"]),
        "--head_size",
        str(dims["head_size"]),
        "--d_decay_lora",
        str(dims["d_decay_lora"]),
        "--d_aaa_lora",
        str(dims["d_aaa_lora"]),
        "--d_mv_lora",
        str(dims["d_mv_lora"]),
        "--d_gate_lora",
        str(dims["d_gate_lora"]),
        "--my_testing",
        os.environ.get("RWKV_SFT_SMOKE_MY_TESTING", "x070"),
        "--kernel",
        os.environ.get("RWKV_SFT_SMOKE_KERNEL", ""),
        "--lr_init",
        "1e-5",
        "--lr_final",
        "1e-5",
        "--warmup_steps",
        "0",
        "--weight_decay",
        "0",
        "--accelerator",
        "gpu",
        "--devices",
        os.environ.get("RWKV_SFT_SMOKE_DEVICES", "1"),
        "--precision",
        "bf16",
        "--strategy",
        strategy,
        "--grad_cp",
        os.environ.get("RWKV_SFT_SMOKE_GRAD_CP", "1"),
        "--enable_progress_bar",
        "False",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("RWKV_SFT_SMOKE_TIMEOUT", "1800")),
    )
    if result.returncode != 0:
        output_tail = (result.stdout + "\n" + result.stderr)[-8000:]
        pytest.fail(f"train.py SFT CUDA smoke failed with code {result.returncode}\n{output_tail}")

    assert (proj_dir / "train_log.txt").is_file()

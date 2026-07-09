import json
import os
import subprocess
import sys
import textwrap
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from src import trainer as trainer_mod
from src.sft_split import compute_sft_shuffled_split_indices

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


def _require_cuda_flag(env_name: str) -> None:
    if os.environ.get(env_name) != "1":
        pytest.skip(f"set {env_name}=1 to run this CUDA SFT smoke test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _load_masked_head_ce_extension(chunk_rows: int):
    from torch.utils.cpp_extension import load

    return load(
        name=f"rwkv7_head_l2wrap_ce_bf16_v4_op_equiv_{chunk_rows}",
        sources=[
            str(ROOT / "cuda" / "rwkv7_head_l2wrap_ce_bf16_v4.cpp"),
            str(ROOT / "cuda" / "rwkv7_head_l2wrap_ce_bf16_v4.cu"),
        ],
        extra_cflags=["-O3", f"-DHEAD_CE_CHUNK={chunk_rows}"],
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-DHEAD_CE_CHUNK={chunk_rows}",
        ],
        verbose=True,
    )


def _build_tiny_sft_binidx(tmp_path: Path, pad_length: int) -> Path:
    from src.sft_binidx import build_binidx_dataset

    prefix = tmp_path / "tiny_sft"
    stats = build_binidx_dataset(
        [str(ROOT / "data" / "SFT" / "sample" / "think.jsonl")],
        output_prefix=str(prefix),
        vocab_path=str(ROOT / "rwkv_vocab_v20260603.txt"),
        template_path=str(ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"),
        n_epoch=1,
        seed=1234,
        pad_length=pad_length,
        num_workers=1,
        shuffle=False,
    )
    assert stats["documents"] == 1
    assert stats["trainable_tokens"] > 0
    return prefix


def _build_accum_equiv_sft_binidx(tmp_path: Path, pad_length: int, docs: int, vocab_size: int) -> Path:
    from src.sft_binidx import EncodedDocument, write_documents

    prefix = tmp_path / "accum_equiv_sft"
    max_token_id = max(32, vocab_size - 16)
    encoded_docs = []
    for doc_id in range(docs):
        input_ids = [10 + ((doc_id * 997 + pos * 17) % (max_token_id - 10)) for pos in range(pad_length)]
        train_tokens = pad_length - 1
        if train_tokens <= 0:
            loss_mask = [0] * pad_length
        elif doc_id % 4 == 0:
            loss_mask = [0] + [1] * train_tokens
        elif doc_id % 4 == 1:
            single_index = 1 + (doc_id % train_tokens)
            loss_mask = [0] + [1 if pos == single_index else 0 for pos in range(1, pad_length)]
        elif doc_id % 4 == 2:
            short_tokens = max(1, min(16, train_tokens // 8))
            loss_mask = [0] + [1 if pos <= short_tokens else 0 for pos in range(1, pad_length)]
        else:
            loss_mask = [0] + [1 if (pos % 3 != 0) else 0 for pos in range(1, pad_length)]
        encoded_docs.append(
            EncodedDocument(
                input_ids=input_ids,
                loss_mask=loss_mask,
            )
        )
    write_documents(str(prefix), encoded_docs)
    return prefix


def _build_synthetic_sft_binidx(tmp_path: Path, pad_length: int, docs: int, vocab_size: int, name: str) -> Path:
    from src.sft_binidx import EncodedDocument, write_documents

    prefix = tmp_path / name
    max_token_id = max(32, vocab_size - 16)
    encoded_docs = []
    for doc_id in range(docs):
        input_ids = [16 + ((doc_id * 131 + pos * 7) % (max_token_id - 16)) for pos in range(pad_length)]
        train_tokens = pad_length - 1
        if train_tokens <= 0:
            loss_mask = [0] * pad_length
        else:
            pattern = doc_id % 6
            if pattern == 0:
                loss_mask = [0] + [1] * train_tokens
            elif pattern == 1:
                single_index = 1 + ((doc_id * 7) % train_tokens)
                loss_mask = [0] + [1 if pos == single_index else 0 for pos in range(1, pad_length)]
            elif pattern == 2:
                short_prefix = max(1, min(16, max(1, train_tokens // 32)))
                loss_mask = [0] + [1 if pos <= short_prefix else 0 for pos in range(1, pad_length)]
            elif pattern == 3:
                medium_prefix = max(1, train_tokens // 4)
                loss_mask = [0] + [1 if pos <= medium_prefix else 0 for pos in range(1, pad_length)]
            elif pattern == 4:
                loss_mask = [0] + [1 if (pos % 2 == 0) else 0 for pos in range(1, pad_length)]
            else:
                stride = 7
                phase = doc_id % stride
                loss_mask = [0] + [1 if ((pos + phase) % stride == 0) else 0 for pos in range(1, pad_length)]
        encoded_docs.append(EncodedDocument(input_ids=input_ids, loss_mask=loss_mask))
    if docs >= 2 and pad_length > 2:
        mask_counts = [sum(doc.loss_mask) for doc in encoded_docs]
        assert min(mask_counts) == 1
        assert max(mask_counts) == pad_length - 1
        assert len(set(mask_counts)) > 1
    write_documents(str(prefix), encoded_docs)
    return prefix


def _load_sft_mask_counts(prefix: Path) -> list[int]:
    from src.binidx import MMapIndexedDataset

    mask_data = MMapIndexedDataset(str(prefix) + ".mask")
    return [int(mask_data[i].sum()) for i in range(len(mask_data))]


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_sft_masked_fused_ce_op_matches_full_logits(tmp_path):
    _require_cuda_flag("RWKV_RUN_CUDA_SFT_MASKED_FUSED_CE_OP_EQUIV_SMOKE")
    if torch.version.hip is not None:
        pytest.skip("SFT masked fused CE op equivalence is CUDA-only")

    batch = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_BATCH", "2"))
    time_len = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_TIME", "2049"))
    hidden_size = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_HIDDEN", "4096"))
    vocab_size = 65536
    chunk_rows_values = [
        int(value.strip())
        for value in os.environ.get("RWKV_SFT_FUSED_CE_OP_CHUNKS", "257,4096,8192").split(",")
        if value.strip()
    ]
    dtype_name = os.environ.get("RWKV_SFT_FUSED_CE_OP_DTYPE", "bf16").lower()
    dtype = torch.float32 if dtype_name in {"fp32", "float32"} else torch.bfloat16
    seed = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_SEED", "1234"))
    assert batch > 0 and time_len > 0 and hidden_size > 0 and chunk_rows_values
    assert all(chunk_rows > 0 for chunk_rows in chunk_rows_values)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    extension = _load_masked_head_ce_extension(max(chunk_rows_values))

    base_hidden = (torch.randn(batch, time_len, hidden_size, device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    base_weight = (torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float32) * 0.02).to(dtype)
    targets = torch.randint(0, vocab_size, (batch, time_len), device="cuda", dtype=torch.long)
    sparse_mask = ((torch.arange(batch * time_len, device="cuda") % 3) != 0).view(batch, time_len).float().contiguous()
    sparse_mask[-1, -1] = 1.0
    single_token_mask = torch.zeros_like(sparse_mask)
    single_token_mask[batch // 2, time_len // 2] = 1.0
    mask_cases = {
        "all_active": torch.ones_like(sparse_mask),
        "sparse_mod3": sparse_mask,
        "single_active": single_token_mask,
        "zero_active": torch.zeros_like(sparse_mask),
    }

    def run_reference(loss_mask: torch.Tensor):
        hidden = base_hidden.detach().clone().requires_grad_(True)
        weight = base_weight.detach().clone().requires_grad_(True)
        logits = F.linear(hidden, weight).float()
        per_token = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        mask_sum = loss_mask.sum()
        if float(mask_sum.detach().item()) > 0:
            loss = (per_token * loss_mask).sum() / mask_sum
        else:
            loss = logits.sum() * 0
        loss.backward()
        return loss.detach(), hidden.grad.detach(), weight.grad.detach()

    def run_fused(loss_mask: torch.Tensor, chunk_rows: int):
        hidden = base_hidden.detach().clone().contiguous()
        weight = base_weight.detach().clone().contiguous()
        loss, grad_hidden, grad_weight = extension.forward_masked(
            hidden,
            weight,
            targets.contiguous(),
            loss_mask.contiguous(),
            chunk_rows,
        )
        torch.cuda.synchronize()
        return loss.detach(), grad_hidden.detach(), grad_weight.detach()

    loss_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_OP_LOSS_ATOL", "2e-2"))
    grad_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_OP_GRAD_ATOL", "2e-2"))
    zero_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_OP_ZERO_ATOL", "1e-7"))
    cases = {}
    max_loss_diff = 0.0
    max_grad_hidden_diff = 0.0
    max_grad_weight_diff = 0.0
    max_zero_abs = 0.0
    for mask_name, loss_mask in mask_cases.items():
        ref_loss, ref_grad_hidden, ref_grad_weight = run_reference(loss_mask)
        chunk_cases = {}
        for chunk_rows in chunk_rows_values:
            fused_loss, fused_grad_hidden, fused_grad_weight = run_fused(loss_mask, chunk_rows)
            loss_diff = float((ref_loss.float() - fused_loss.float()).abs().item())
            grad_hidden_diff = float((ref_grad_hidden.float() - fused_grad_hidden.float()).abs().max().item())
            grad_weight_diff = float((ref_grad_weight.float() - fused_grad_weight.float()).abs().max().item())
            grad_hidden_mean_diff = float((ref_grad_hidden.float() - fused_grad_hidden.float()).abs().mean().item())
            grad_weight_mean_diff = float((ref_grad_weight.float() - fused_grad_weight.float()).abs().mean().item())
            max_loss_diff = max(max_loss_diff, loss_diff)
            max_grad_hidden_diff = max(max_grad_hidden_diff, grad_hidden_diff)
            max_grad_weight_diff = max(max_grad_weight_diff, grad_weight_diff)
            if mask_name == "zero_active":
                max_zero_abs = max(max_zero_abs, float(fused_loss.float().abs().item()))
                max_zero_abs = max(max_zero_abs, float(fused_grad_hidden.float().abs().max().item()))
                max_zero_abs = max(max_zero_abs, float(fused_grad_weight.float().abs().max().item()))
            chunk_cases[str(chunk_rows)] = {
                "reference_loss": float(ref_loss.float().item()),
                "fused_loss": float(fused_loss.float().item()),
                "loss_diff": loss_diff,
                "grad_hidden_max_abs_diff": grad_hidden_diff,
                "grad_weight_max_abs_diff": grad_weight_diff,
                "grad_hidden_mean_abs_diff": grad_hidden_mean_diff,
                "grad_weight_mean_abs_diff": grad_weight_mean_diff,
            }
        cases[mask_name] = {
            "mask_sum": float(loss_mask.sum().item()),
            "chunks": chunk_cases,
        }

    summary = {
        "batch": batch,
        "time_len": time_len,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "chunk_rows_values": chunk_rows_values,
        "dtype": str(dtype).removeprefix("torch."),
        "seed": seed,
        "cases": cases,
        "max_loss_diff": max_loss_diff,
        "loss_atol": loss_atol,
        "max_grad_hidden_diff": max_grad_hidden_diff,
        "max_grad_weight_diff": max_grad_weight_diff,
        "grad_atol": grad_atol,
        "max_zero_abs": max_zero_abs,
        "zero_atol": zero_atol,
    }
    summary_file = os.environ.get("RWKV_SFT_FUSED_CE_OP_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert max_loss_diff <= loss_atol, json.dumps(summary, indent=2)
    assert max_grad_hidden_diff <= grad_atol, json.dumps(summary, indent=2)
    assert max_grad_weight_diff <= grad_atol, json.dumps(summary, indent=2)
    assert max_zero_abs <= zero_atol, json.dumps(summary, indent=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_sft_masked_fused_ce_op_longrow_matches_full_logits(tmp_path):
    _require_cuda_flag("RWKV_RUN_CUDA_SFT_MASKED_FUSED_CE_OP_LONGROW_SMOKE")
    if torch.version.hip is not None:
        pytest.skip("SFT masked fused CE long-row equivalence is CUDA-only")

    batch = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_BATCH", "1"))
    time_len = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_TIME", "86017"))
    hidden_size = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_HIDDEN", "128"))
    vocab_size = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_VOCAB", "65536"))
    chunk_rows = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_CHUNK", "257"))
    ref_row_chunk = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_REF_ROW_CHUNK", "2048"))
    dtype_name = os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_DTYPE", "bf16").lower()
    dtype = torch.float32 if dtype_name in {"fp32", "float32"} else torch.bfloat16
    seed = int(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_SEED", "1234"))
    assert batch > 0 and time_len > 0 and hidden_size > 0 and vocab_size > 0
    assert chunk_rows > 0 and ref_row_chunk > 0
    if vocab_size != 65536:
        pytest.skip("rwkv7_head_l2wrap_ce_bf16_v4.forward_masked currently expects vocab_size=65536")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    extension = _load_masked_head_ce_extension(chunk_rows)

    base_hidden = (torch.randn(batch, time_len, hidden_size, device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    base_weight = (torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float32) * 0.02).to(dtype)
    targets = torch.randint(0, vocab_size, (batch, time_len), device="cuda", dtype=torch.long)
    loss_mask = torch.ones(batch, time_len, device="cuda", dtype=torch.float32).contiguous()

    def run_reference():
        flat_hidden = base_hidden.detach().float().view(-1, hidden_size)
        flat_targets = targets.reshape(-1)
        flat_mask = loss_mask.reshape(-1).float()
        mask_sum = flat_mask.sum()
        assert float(mask_sum.item()) > 0
        grad_hidden = torch.zeros_like(flat_hidden)
        grad_weight = torch.zeros(vocab_size, hidden_size, device="cuda", dtype=torch.float32)
        loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)

        for row_start in range(0, flat_hidden.shape[0], ref_row_chunk):
            row_end = min(row_start + ref_row_chunk, flat_hidden.shape[0])
            hidden_chunk = flat_hidden[row_start:row_end].clone().requires_grad_(True)
            weight = base_weight.detach().float().clone().requires_grad_(True)
            target_chunk = flat_targets[row_start:row_end]
            mask_chunk = flat_mask[row_start:row_end]
            logits = F.linear(hidden_chunk, weight)
            per_token = F.cross_entropy(logits, target_chunk, reduction="none")
            loss_chunk = (per_token * mask_chunk).sum() / mask_sum
            loss_chunk.backward()
            grad_hidden[row_start:row_end] = hidden_chunk.grad.detach()
            grad_weight += weight.grad.detach()
            loss_sum += loss_chunk.detach()

        return (
            loss_sum.detach(),
            grad_hidden.view(batch, time_len, hidden_size).detach(),
            grad_weight.detach(),
        )

    def run_fused():
        hidden = base_hidden.detach().clone().contiguous()
        weight = base_weight.detach().clone().contiguous()
        loss, grad_hidden, grad_weight = extension.forward_masked(
            hidden,
            weight,
            targets.contiguous(),
            loss_mask,
            chunk_rows,
        )
        torch.cuda.synchronize()
        return loss.detach(), grad_hidden.detach(), grad_weight.detach()

    ref_loss, ref_grad_hidden, ref_grad_weight = run_reference()
    fused_loss, fused_grad_hidden, fused_grad_weight = run_fused()

    loss_diff = float((ref_loss.float() - fused_loss.float()).abs().item())
    grad_hidden_max_abs_diff = float((ref_grad_hidden.float() - fused_grad_hidden.float()).abs().max().item())
    grad_weight_max_abs_diff = float((ref_grad_weight.float() - fused_grad_weight.float()).abs().max().item())
    grad_hidden_mean_abs_diff = float((ref_grad_hidden.float() - fused_grad_hidden.float()).abs().mean().item())
    grad_weight_mean_abs_diff = float((ref_grad_weight.float() - fused_grad_weight.float()).abs().mean().item())
    loss_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_LOSS_ATOL", "2e-2"))
    grad_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_GRAD_ATOL", "3e-2"))
    mean_grad_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_MEAN_GRAD_ATOL", "5e-3"))
    summary = {
        "batch": batch,
        "time_len": time_len,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "chunk_rows": chunk_rows,
        "ref_row_chunk": ref_row_chunk,
        "dtype": str(dtype).removeprefix("torch."),
        "seed": seed,
        "mask_sum": float(loss_mask.sum().item()),
        "loss_diff": loss_diff,
        "loss_atol": loss_atol,
        "grad_hidden_max_abs_diff": grad_hidden_max_abs_diff,
        "grad_weight_max_abs_diff": grad_weight_max_abs_diff,
        "grad_atol": grad_atol,
        "grad_hidden_mean_abs_diff": grad_hidden_mean_abs_diff,
        "grad_weight_mean_abs_diff": grad_weight_mean_abs_diff,
        "mean_grad_atol": mean_grad_atol,
    }
    summary_file = os.environ.get("RWKV_SFT_FUSED_CE_OP_LONGROW_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert loss_diff <= loss_atol, json.dumps(summary, indent=2)
    assert grad_hidden_max_abs_diff <= grad_atol, json.dumps(summary, indent=2)
    assert grad_weight_max_abs_diff <= grad_atol, json.dumps(summary, indent=2)
    assert grad_hidden_mean_abs_diff <= mean_grad_atol, json.dumps(summary, indent=2)
    assert grad_weight_mean_abs_diff <= mean_grad_atol, json.dumps(summary, indent=2)


def _base_sft_args(prefix: Path, dims: dict[str, int], ctx_len: int, **overrides) -> SimpleNamespace:
    args = SimpleNamespace(
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
        accumulate_grad_batches=1,
        sft_masked_ce_chunk=0,
        sft_masked_fused_ce_chunk=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _train_py_command(
    *,
    load_model: Path,
    prefix: Path,
    proj_dir: Path,
    dims: dict[str, int],
    ctx_len: int,
    epoch_steps: int,
    epoch_count: int,
    micro_bsz: int = 1,
    sft_masked_fused_ce_chunk: int | None = None,
    accumulate_grad_batches: int | None = None,
    devices: int | str | None = None,
    strategy: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    if accumulate_grad_batches is None:
        accumulate_grad_batches = int(os.environ.get("RWKV_SFT_SMOKE_ACCUMULATE_GRAD_BATCHES", "1"))
    if sft_masked_fused_ce_chunk is None:
        sft_masked_fused_ce_chunk = int(os.environ.get("RWKV_SFT_SMOKE_MASKED_FUSED_CE_CHUNK", "0"))
    if devices is None:
        devices = os.environ.get("RWKV_SFT_SMOKE_DEVICES", "1")
    if strategy is None:
        strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_2")

    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--load_model",
        str(load_model),
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
        str(epoch_steps),
        "--epoch_count",
        str(epoch_count),
        "--sft_one_pass",
        "0",
        "--sft_masked_fused_ce_chunk",
        str(sft_masked_fused_ce_chunk),
        "--micro_bsz",
        str(micro_bsz),
        "--accumulate_grad_batches",
        str(accumulate_grad_batches),
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
        str(devices),
        "--precision",
        "bf16",
        "--strategy",
        str(strategy),
        "--grad_cp",
        os.environ.get("RWKV_SFT_SMOKE_GRAD_CP", "1"),
        "--enable_progress_bar",
        "False",
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def _read_train_log_epoch_loss(proj_dir: Path) -> float:
    log_path = proj_dir / "train_log.txt"
    assert log_path.is_file(), f"missing train log: {log_path}"
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, f"empty train log: {log_path}"
    parts = lines[-1].split()
    assert len(parts) >= 2, f"unexpected train log line: {lines[-1]}"
    return float(parts[1])


def _read_train_log_last_lr(proj_dir: Path) -> float:
    log_path = proj_dir / "train_log.txt"
    assert log_path.is_file(), f"missing train log: {log_path}"
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    epoch_lines = [line for line in lines if line[0].isdigit()]
    assert epoch_lines, f"missing epoch lines in train log: {log_path}"
    parts = epoch_lines[-1].split()
    assert len(parts) >= 4, f"unexpected train log line: {epoch_lines[-1]}"
    return float(parts[3])


def _assert_step_checkpoint_exists(proj_dir: Path) -> None:
    step_checkpoints = sorted(proj_dir.glob("rwkv-step-*.pth"))
    if step_checkpoints:
        return
    entries = sorted(path.name for path in proj_dir.iterdir()) if proj_dir.exists() else []
    raise AssertionError(f"missing rwkv-step-*.pth checkpoint in {proj_dir}; entries={entries}")


def _run_train_py(command: list[str], label: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("RWKV_SFT_SMOKE_TIMEOUT", "1800")),
    )
    combined_output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        pytest.fail(f"{label} failed with code {result.returncode}\n{combined_output[-8000:]}")
    return combined_output


def _write_train_py_sft_gather_verify_wrapper(script_path: Path) -> Path:
    script_path.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import json
            import os
            import runpy
            import sys
            from pathlib import Path

            ROOT = Path({str(ROOT)!r})
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            def _rank0_log_event(payload):
                log_path = os.environ.get("RWKV_SFT_GATHER_VERIFY_LOG", "")
                if not log_path:
                    return
                if int(os.environ.get("RANK", "0") or 0) != 0:
                    return
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\\n")

            def _apply_train_env_from_argv(argv):
                parser = argparse.ArgumentParser(add_help=False)
                parser.add_argument("--head_chunk", default="0")
                parser.add_argument("--sft_masked_fused_ce_chunk", default="0")
                parser.add_argument("--head_size", default="64")
                parser.add_argument("--my_testing", default="")
                parser.add_argument("--kernel", default="")
                parser.add_argument("--precision", default="bf16")
                parser.add_argument("--strategy", default="")
                known, _ = parser.parse_known_args(argv)
                os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = str(known.head_chunk)
                os.environ["RWKV_SFT_MASKED_FUSED_CE_CHUNK"] = str(known.sft_masked_fused_ce_chunk)
                os.environ["RWKV_HEAD_SIZE"] = str(known.head_size)
                os.environ["RWKV_MY_TESTING"] = str(known.my_testing)
                os.environ["RWKV_KERNEL"] = str(known.kernel)
                os.environ["RWKV_FLOAT_MODE"] = str(known.precision)
                os.environ["RWKV_JIT_ON"] = "0" if "deepspeed_stage_3" in str(known.strategy) else "1"

            _apply_train_env_from_argv(sys.argv[1:])

            import torch
            import torch.distributed as dist
            from pytorch_lightning.utilities import rank_zero_info
            from src.model import RWKV, head_masked_cross_entropy_cuda
            from src.trainer import train_callback

            _ORIG_SFT_LOSS_STEP = RWKV.sft_loss_step
            _ORIG_ON_BEFORE_OPTIMIZER_STEP = train_callback.on_before_optimizer_step

            def _patched_sft_loss_step(self, batch, batch_idx, gather_head=False):
                loss = _ORIG_SFT_LOSS_STEP(self, batch, batch_idx, gather_head=gather_head)
                if (not gather_head) and isinstance(batch, (tuple, list)) and len(batch) == 3:
                    self._sft_verify_last_batch = tuple(t.detach().clone() for t in batch)
                    self._sft_verify_last_batch_idx = int(batch_idx)
                    self._sft_verify_last_local_loss = loss.detach().float()
                    _rank0_log_event(
                        {{
                            "event": "captured_batch",
                            "batch_idx": int(batch_idx),
                            "local_mask_tokens": float(batch[2].detach().float().sum().item()),
                        }}
                    )
                return loss

            def _patched_on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=None):
                _ORIG_ON_BEFORE_OPTIMIZER_STEP(self, trainer, pl_module, optimizer, optimizer_idx)
                batch = getattr(pl_module, "_sft_verify_last_batch", None)
                train_loss = getattr(pl_module, "_sft_verify_last_local_loss", None)
                if batch is None or train_loss is None:
                    _rank0_log_event(
                        {{
                            "event": "missing_capture",
                            "global_step": int(getattr(trainer, "global_step", 0)),
                        }}
                    )
                    return

                batch_idx = int(getattr(pl_module, "_sft_verify_last_batch_idx", 0))
                local_tokens = batch[2].detach().float().sum().to(device=train_loss.device, dtype=torch.float32)
                fused_chunk = int(getattr(pl_module.args, "sft_masked_fused_ce_chunk", 0) or 0)
                if fused_chunk <= 0:
                    raise AssertionError("SFT gather verify wrapper expects sft_masked_fused_ce_chunk > 0.")

                def _gather_full_weight(linear_weight, device):
                    ds_tensor = getattr(linear_weight, "ds_tensor", None)
                    ds_shape = tuple(getattr(linear_weight, "ds_shape", linear_weight.shape))
                    ds_numel = int(getattr(linear_weight, "ds_numel", linear_weight.numel()))
                    if ds_tensor is None or not (dist.is_available() and dist.is_initialized()):
                        return linear_weight.detach().to(device=device).view(ds_shape)
                    shard = ds_tensor.detach().to(device=device)
                    gathered = [torch.empty_like(shard) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered, shard)
                    return torch.cat(gathered, dim=0)[:ds_numel].view(ds_shape)

                was_training = pl_module.training
                try:
                    pl_module.eval()
                    with torch.no_grad():
                        idx, targets, loss_mask = batch
                        hidden = pl_module._forward_features(idx)
                        full_weight = _gather_full_weight(pl_module.head.weight, hidden.device)
                        ref_loss = head_masked_cross_entropy_cuda(
                            hidden,
                            full_weight,
                            targets,
                            loss_mask,
                            fused_chunk,
                        ).detach().float()
                finally:
                    if was_training:
                        pl_module.train()

                stats = torch.stack(
                    [
                        train_loss.to(dtype=torch.float32) * local_tokens,
                        ref_loss.to(dtype=torch.float32) * local_tokens,
                        local_tokens,
                    ]
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

                token_count = float(stats[2].item())
                train_weighted_loss = float((stats[0] / stats[2]).item()) if token_count > 0 else 0.0
                ref_weighted_loss = float((stats[1] / stats[2]).item()) if token_count > 0 else 0.0
                diff = abs(train_weighted_loss - ref_weighted_loss)
                tol = float(os.environ.get("RWKV_SFT_GATHER_VERIFY_TOL", "5e-4"))

                pl_module._sft_verify_last_batch = None
                pl_module._sft_verify_last_batch_idx = None
                pl_module._sft_verify_last_local_loss = None

                if trainer.is_global_zero:
                    _rank0_log_event(
                        {{
                            "event": "verified",
                            "batch_idx": batch_idx,
                            "train_weighted_loss": train_weighted_loss,
                            "ref_weighted_loss": ref_weighted_loss,
                            "abs_diff": diff,
                            "tol": tol,
                            "global_mask_tokens": token_count,
                        }}
                    )
                    rank_zero_info(
                        "########## SFT_GATHER_VERIFY "
                        f"train_loss={{train_weighted_loss:.6f}} "
                        f"ref_loss={{ref_weighted_loss:.6f}} "
                        f"diff={{diff:.6e}} tokens={{token_count:.0f}} ##########"
                    )

                if diff > tol:
                    raise AssertionError(
                        json.dumps(
                            {{
                                "train_weighted_loss": train_weighted_loss,
                                "ref_weighted_loss": ref_weighted_loss,
                                "abs_diff": diff,
                                "tol": tol,
                                "global_mask_tokens": token_count,
                                "batch_idx": batch_idx,
                            }},
                            indent=2,
                        )
                    )

            RWKV.sft_loss_step = _patched_sft_loss_step
            train_callback.on_before_optimizer_step = _patched_on_before_optimizer_step

            sys.argv[0] = str(ROOT / "train.py")
            runpy.run_path(sys.argv[0], run_name="__main__")
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return script_path


def _fake_wandb_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_dir = tmp_path / "fake_wandb_module"
    fake_dir.mkdir()
    fake_log = tmp_path / "fake_wandb.jsonl"
    (fake_dir / "wandb.py").write_text(
        "\n".join(
            [
                "import json, os",
                "_log_path = os.environ.get('RWKV_FAKE_WANDB_LOG', '')",
                "def _write(payload):",
                "    if _log_path:",
                "        with open(_log_path, 'a', encoding='utf-8') as f:",
                "            f.write(json.dumps(payload, default=str, sort_keys=True) + '\\n')",
                "def init(**kwargs):",
                "    _write({'event': 'init', 'kwargs': kwargs})",
                "    return None",
                "def log(values, step=None):",
                "    _write({'event': 'log', 'step': step, 'values': values})",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["RWKV_FAKE_WANDB_LOG"] = str(fake_log)
    return env, fake_log


def _read_fake_wandb_train_losses(fake_log: Path) -> dict[int, float]:
    losses: dict[int, float] = {}
    for line in fake_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "log":
            continue
        values = event.get("values", {})
        if "train/loss" in values:
            losses[int(event["step"])] = float(values["train/loss"])
    return losses


def _run_checkpoint_merge_command(command: list[str], label: str) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("RWKV_SFT_MERGE_TIMEOUT", os.environ.get("RWKV_SFT_SMOKE_TIMEOUT", "3600"))),
    )
    combined_output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        pytest.fail(f"{label} failed with code {result.returncode}\n{combined_output[-12000:]}")
    return combined_output


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


def test_build_tiny_sft_binidx_smoke_input(tmp_path):
    from src.binidx import MMapIndexedDataset

    prefix = _build_tiny_sft_binidx(tmp_path, 257)
    token_data = MMapIndexedDataset(str(prefix))
    mask_data = MMapIndexedDataset(str(prefix) + ".mask")

    assert len(token_data) == 1
    assert len(mask_data) == 1
    assert int(token_data.sizes[0]) == 257
    assert int(mask_data.sizes[0]) == 257
    assert int(mask_data[0].sum()) > 0


def test_build_synthetic_sft_binidx_uses_high_variance_loss_masks(tmp_path):
    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length=257,
        docs=12,
        vocab_size=512,
        name="synthetic_variance_sft",
    )
    mask_counts = _load_sft_mask_counts(prefix)

    assert min(mask_counts) == 1
    assert max(mask_counts) == 256
    assert len(set(mask_counts)) >= 6
    assert max(mask_counts) / min(mask_counts) >= 256


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
    monkeypatch.setenv("RWKV_SFT_MASKED_FUSED_CE_CHUNK", os.environ.get("RWKV_SFT_SMOKE_MASKED_FUSED_CE_CHUNK", "0"))
    monkeypatch.setenv("RWKV_FLOAT_MODE", "bf16")

    from src.dataset import MyDataset
    from src.model import RWKV

    model_args = SimpleNamespace(
        **dims,
        dim_att=dims["n_embd"],
        my_testing=os.environ["RWKV_MY_TESTING"],
        grad_cp=0,
        ctx_len=ctx_len,
        sft_masked_ce_chunk=int(os.environ.get("RWKV_SFT_SMOKE_MASKED_CE_CHUNK", "0")),
        sft_masked_fused_ce_chunk=int(os.environ.get("RWKV_SFT_SMOKE_MASKED_FUSED_CE_CHUNK", "0")),
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
def test_cuda_sft_gradient_accumulation_loss_matches_large_batch(tmp_path, monkeypatch):
    model_path = _require_cuda_smoke("RWKV_RUN_CUDA_SFT_ACCUM_EQUIV_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_ACCUM_EQUIV_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    accum_steps = int(os.environ.get("RWKV_SFT_ACCUM_EQUIV_STEPS", "2"))
    assert accum_steps >= 2
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    prefix = _build_accum_equiv_sft_binidx(tmp_path, pad_length, accum_steps, dims["vocab_size"])

    monkeypatch.setenv("RWKV_MY_TESTING", os.environ.get("RWKV_SFT_SMOKE_MY_TESTING", "x070"))
    monkeypatch.setenv("RWKV_KERNEL", os.environ.get("RWKV_SFT_SMOKE_KERNEL", ""))
    monkeypatch.setenv("RWKV_JIT_ON", "1")
    monkeypatch.setenv("RWKV_CTXLEN", str(ctx_len))
    monkeypatch.setenv("RWKV_HEAD_SIZE", str(dims["head_size"]))
    monkeypatch.setenv("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    monkeypatch.setenv("RWKV_SFT_MASKED_FUSED_CE_CHUNK", os.environ.get("RWKV_SFT_SMOKE_MASKED_FUSED_CE_CHUNK", "0"))
    monkeypatch.setenv("RWKV_FLOAT_MODE", "bf16")

    from src.dataset import MyDataset
    from src.model import RWKV

    model_args = SimpleNamespace(
        **dims,
        dim_att=dims["n_embd"],
        my_testing=os.environ["RWKV_MY_TESTING"],
        grad_cp=0,
        ctx_len=ctx_len,
        sft_masked_ce_chunk=int(os.environ.get("RWKV_SFT_SMOKE_MASKED_CE_CHUNK", "0")),
        sft_masked_fused_ce_chunk=int(os.environ.get("RWKV_SFT_SMOKE_MASKED_FUSED_CE_CHUNK", "0")),
    )
    model = RWKV(model_args).to(device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = MyDataset(_base_sft_args(prefix, dims, ctx_len, epoch_steps=accum_steps))
    examples = [dataset[idx] for idx in range(accum_steps)]
    large_batch = tuple(torch.stack([example[field] for example in examples]).cuda(non_blocking=True) for field in range(3))

    with torch.no_grad():
        large_batch_loss_sum = model.training_step(large_batch, 0).detach().float()
        large_batch_mask_count = large_batch[2].sum().detach().float()
        accumulated_loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        mask_count_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        mask_counts = []
        micro_loss_sums = []

        for example in examples:
            micro_batch = tuple(t.unsqueeze(0).cuda(non_blocking=True) for t in example)
            micro_loss_sum = model.training_step(micro_batch, 0).detach().float()
            mask_count = micro_batch[2].sum().detach().float()
            assert mask_count.item() > 0
            accumulated_loss_sum += micro_loss_sum
            mask_count_sum += mask_count
            mask_counts.append(mask_count.item())
            micro_loss_sums.append(micro_loss_sum.item())

        large_batch_loss = large_batch_loss_sum / large_batch_mask_count
        weighted_accumulated_loss = accumulated_loss_sum / mask_count_sum

    loss_value = large_batch_loss.item()
    weighted_value = weighted_accumulated_loss.item()
    weighted_diff = abs(loss_value - weighted_value)
    atol = float(os.environ.get("RWKV_SFT_ACCUM_EQUIV_ATOL", "1e-2"))
    rtol = float(os.environ.get("RWKV_SFT_ACCUM_EQUIV_RTOL", "1e-3"))
    allowed = atol + rtol * abs(loss_value)
    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "accum_steps": accum_steps,
        "large_batch_loss": loss_value,
        "token_weighted_accumulated_loss": weighted_value,
        "token_weighted_diff": weighted_diff,
        "allowed_diff": allowed,
        "mask_counts": mask_counts,
        "micro_loss_sums": micro_loss_sums,
    }
    summary_file = os.environ.get("RWKV_SFT_ACCUM_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert len(set(mask_counts)) > 1, json.dumps(summary, indent=2)
    assert weighted_diff <= allowed, json.dumps(summary, indent=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_sft_masked_ce_chunk_training_matches_full_logits(tmp_path, monkeypatch):
    model_path = _require_cuda_smoke("RWKV_RUN_CUDA_SFT_MASKED_CE_CHUNK_EQUIV_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_CHUNK_EQUIV_PAD_LENGTH", "1025"))
    ctx_len = pad_length - 1
    steps = int(os.environ.get("RWKV_SFT_CHUNK_EQUIV_STEPS", "4"))
    micro_bsz = int(os.environ.get("RWKV_SFT_CHUNK_EQUIV_MICRO_BSZ", "1"))
    chunk_size = int(os.environ.get("RWKV_SFT_CHUNK_EQUIV_CHUNK", "512"))
    lr = float(os.environ.get("RWKV_SFT_CHUNK_EQUIV_LR", "1e-5"))
    assert steps >= 1
    assert micro_bsz >= 1
    assert chunk_size > 0
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    prefix = _build_accum_equiv_sft_binidx(tmp_path, pad_length, docs=steps * micro_bsz, vocab_size=dims["vocab_size"])

    monkeypatch.setenv("RWKV_MY_TESTING", os.environ.get("RWKV_SFT_SMOKE_MY_TESTING", "x070"))
    monkeypatch.setenv("RWKV_KERNEL", os.environ.get("RWKV_SFT_SMOKE_KERNEL", ""))
    monkeypatch.setenv("RWKV_JIT_ON", "1")
    monkeypatch.setenv("RWKV_CTXLEN", str(ctx_len))
    monkeypatch.setenv("RWKV_HEAD_SIZE", str(dims["head_size"]))
    monkeypatch.setenv("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    monkeypatch.setenv("RWKV_SFT_MASKED_FUSED_CE_CHUNK", "0")
    monkeypatch.setenv("RWKV_FLOAT_MODE", "bf16")

    from src.dataset import MyDataset
    from src.model import RWKV

    dataset = MyDataset(_base_sft_args(prefix, dims, ctx_len, epoch_steps=steps, micro_bsz=micro_bsz))
    examples = [dataset[idx] for idx in range(steps * micro_bsz)]
    selected_names = [
        "emb.weight",
        "head.weight",
        "ln_out.weight",
        "blocks.0.ffn.key.weight",
        "blocks.0.att.receptance.weight",
    ]

    def tensor_snapshot(tensor: torch.Tensor) -> torch.Tensor:
        slices = tuple(slice(0, min(256, dim)) for dim in tensor.shape)
        return tensor.detach().float()[slices].cpu().clone()

    def run_variant(sft_masked_ce_chunk: int):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_args = SimpleNamespace(
            **dims,
            dim_att=dims["n_embd"],
            my_testing=os.environ["RWKV_MY_TESTING"],
            grad_cp=0,
            ctx_len=ctx_len,
            sft_masked_ce_chunk=sft_masked_ce_chunk,
            sft_masked_fused_ce_chunk=0,
        )
        model = RWKV(model_args).to(device="cuda", dtype=torch.bfloat16)
        model.load_state_dict(state, strict=True)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        losses = []
        grad_norms = []
        started_at = time.perf_counter()

        for step in range(steps):
            batch_examples = examples[step * micro_bsz : (step + 1) * micro_bsz]
            batch = tuple(
                torch.stack([example[field] for example in batch_examples]).cuda(non_blocking=True)
                for field in range(3)
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_step(batch, step)
            assert torch.isfinite(loss.detach()).item()
            loss.backward()
            trainer_mod.rescale_sft_token_weighted_gradients(SimpleNamespace(world_size=1), model)
            grad_norm_sq = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_norm_sq += float(parameter.grad.detach().float().norm().item() ** 2)
            grad_norms.append(grad_norm_sq ** 0.5)
            optimizer.step()
            losses.append(float(loss.detach().float().cpu().item()))

        torch.cuda.synchronize()
        elapsed_sec = time.perf_counter() - started_at
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        with torch.no_grad():
            state_dict = model.state_dict()
            snapshots = {
                name: tensor_snapshot(state_dict[name])
                for name in selected_names
                if name in state_dict
            }
            checksum = 0.0
            sq_norm = 0.0
            max_abs = 0.0
            for parameter in model.parameters():
                value = parameter.detach().float()
                checksum += float(value.sum().item())
                sq_norm += float(value.square().sum().item())
                max_abs = max(max_abs, float(value.abs().max().item()))
        del model
        del optimizer
        torch.cuda.empty_cache()
        return {
            "losses": losses,
            "grad_norms": grad_norms,
            "elapsed_sec": elapsed_sec,
            "tokens_per_sec": float(steps * micro_bsz * ctx_len / elapsed_sec),
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_gib": float(peak_memory_bytes / (1024**3)),
            "param_checksum": checksum,
            "param_l2_norm": sq_norm ** 0.5,
            "param_max_abs": max_abs,
            "snapshots": snapshots,
        }

    full = run_variant(0)
    chunked = run_variant(chunk_size)

    loss_diffs = [abs(a - b) for a, b in zip(full["losses"], chunked["losses"])]
    grad_norm_diffs = [abs(a - b) for a, b in zip(full["grad_norms"], chunked["grad_norms"])]
    param_diffs = {
        name: float((full["snapshots"][name] - chunked["snapshots"][name]).abs().max().item())
        for name in full["snapshots"].keys() & chunked["snapshots"].keys()
    }
    max_loss_diff = max(loss_diffs) if loss_diffs else 0.0
    max_grad_norm_diff = max(grad_norm_diffs) if grad_norm_diffs else 0.0
    max_param_diff = max(param_diffs.values()) if param_diffs else 0.0
    checksum_diff = abs(full["param_checksum"] - chunked["param_checksum"])
    l2_norm_diff = abs(full["param_l2_norm"] - chunked["param_l2_norm"])
    loss_atol = float(os.environ.get("RWKV_SFT_CHUNK_EQUIV_LOSS_ATOL", "2e-2"))
    loss_rtol = float(os.environ.get("RWKV_SFT_CHUNK_EQUIV_LOSS_RTOL", "2e-3"))
    param_atol = float(os.environ.get("RWKV_SFT_CHUNK_EQUIV_PARAM_ATOL", "2e-2"))
    param_rtol = float(os.environ.get("RWKV_SFT_CHUNK_EQUIV_PARAM_RTOL", "1e-2"))
    loss_allowed = max(
        loss_atol + loss_rtol * max(abs(a), abs(b))
        for a, b in zip(full["losses"], chunked["losses"])
    )
    param_allowed = param_atol + param_rtol * max(full["param_max_abs"], chunked["param_max_abs"])
    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "steps": steps,
        "micro_bsz": micro_bsz,
        "chunk_size": chunk_size,
        "lr": lr,
        "full_logits": {key: value for key, value in full.items() if key != "snapshots"},
        "chunked_masked_head": {key: value for key, value in chunked.items() if key != "snapshots"},
        "loss_diffs": loss_diffs,
        "max_loss_diff": max_loss_diff,
        "loss_allowed": loss_allowed,
        "grad_norm_diffs": grad_norm_diffs,
        "max_grad_norm_diff": max_grad_norm_diff,
        "param_snapshot_max_abs_diffs": param_diffs,
        "max_param_snapshot_diff": max_param_diff,
        "param_allowed": param_allowed,
        "param_checksum_diff": checksum_diff,
        "param_l2_norm_diff": l2_norm_diff,
        "memory_delta_gib": full["peak_memory_gib"] - chunked["peak_memory_gib"],
        "chunk_peak_memory_ratio": chunked["peak_memory_bytes"] / max(full["peak_memory_bytes"], 1),
        "chunk_elapsed_ratio": chunked["elapsed_sec"] / max(full["elapsed_sec"], 1e-9),
    }
    summary_file = os.environ.get("RWKV_SFT_CHUNK_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert max_loss_diff <= loss_allowed, json.dumps(summary, indent=2)
    assert max_param_diff <= param_allowed, json.dumps(summary, indent=2)
    if os.environ.get("RWKV_SFT_CHUNK_EQUIV_REQUIRE_MEMORY_IMPROVEMENT", "") == "1":
        assert chunked["peak_memory_bytes"] < full["peak_memory_bytes"], json.dumps(summary, indent=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_sft_masked_fused_ce_training_matches_full_logits(tmp_path, monkeypatch):
    model_path = _require_cuda_smoke("RWKV_RUN_CUDA_SFT_MASKED_FUSED_CE_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_FUSED_CE_PAD_LENGTH", "1025"))
    ctx_len = pad_length - 1
    steps = int(os.environ.get("RWKV_SFT_FUSED_CE_STEPS", "4"))
    micro_bsz = int(os.environ.get("RWKV_SFT_FUSED_CE_MICRO_BSZ", "1"))
    fused_chunk = int(os.environ.get("RWKV_SFT_FUSED_CE_CHUNK", "512"))
    lr = float(os.environ.get("RWKV_SFT_FUSED_CE_LR", "1e-5"))
    assert steps >= 1
    assert micro_bsz >= 1
    assert fused_chunk > 0
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    prefix = _build_accum_equiv_sft_binidx(tmp_path, pad_length, docs=steps * micro_bsz, vocab_size=dims["vocab_size"])

    monkeypatch.setenv("RWKV_MY_TESTING", os.environ.get("RWKV_SFT_SMOKE_MY_TESTING", "x070"))
    monkeypatch.setenv("RWKV_KERNEL", os.environ.get("RWKV_SFT_SMOKE_KERNEL", ""))
    monkeypatch.setenv("RWKV_JIT_ON", "1")
    monkeypatch.setenv("RWKV_CTXLEN", str(ctx_len))
    monkeypatch.setenv("RWKV_HEAD_SIZE", str(dims["head_size"]))
    monkeypatch.setenv("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    monkeypatch.setenv("RWKV_SFT_MASKED_FUSED_CE_CHUNK", str(fused_chunk))
    monkeypatch.setenv("RWKV_FLOAT_MODE", "bf16")

    sys.modules.pop("src.model", None)
    from src.dataset import MyDataset
    from src.model import RWKV

    dataset = MyDataset(_base_sft_args(prefix, dims, ctx_len, epoch_steps=steps, micro_bsz=micro_bsz))
    examples = [dataset[idx] for idx in range(steps * micro_bsz)]
    selected_names = [
        "emb.weight",
        "head.weight",
        "ln_out.weight",
        "blocks.0.ffn.key.weight",
        "blocks.0.att.receptance.weight",
    ]

    def tensor_snapshot(tensor: torch.Tensor) -> torch.Tensor:
        slices = tuple(slice(0, min(256, dim)) for dim in tensor.shape)
        return tensor.detach().float()[slices].cpu().clone()

    def run_variant(sft_masked_fused_ce_chunk: int):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_args = SimpleNamespace(
            **dims,
            dim_att=dims["n_embd"],
            my_testing=os.environ["RWKV_MY_TESTING"],
            grad_cp=0,
            ctx_len=ctx_len,
            sft_masked_ce_chunk=0,
            sft_masked_fused_ce_chunk=sft_masked_fused_ce_chunk,
        )
        model = RWKV(model_args).to(device="cuda", dtype=torch.bfloat16)
        model.load_state_dict(state, strict=True)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        losses = []
        grad_norms = []
        started_at = time.perf_counter()

        for step in range(steps):
            batch_examples = examples[step * micro_bsz : (step + 1) * micro_bsz]
            batch = tuple(
                torch.stack([example[field] for example in batch_examples]).cuda(non_blocking=True)
                for field in range(3)
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_step(batch, step)
            assert torch.isfinite(loss.detach()).item()
            loss.backward()
            trainer_mod.rescale_sft_token_weighted_gradients(SimpleNamespace(world_size=1), model)
            grad_norm_sq = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_norm_sq += float(parameter.grad.detach().float().norm().item() ** 2)
            grad_norms.append(grad_norm_sq ** 0.5)
            optimizer.step()
            losses.append(float(loss.detach().float().cpu().item()))

        torch.cuda.synchronize()
        elapsed_sec = time.perf_counter() - started_at
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        with torch.no_grad():
            state_dict = model.state_dict()
            snapshots = {
                name: tensor_snapshot(state_dict[name])
                for name in selected_names
                if name in state_dict
            }
            checksum = 0.0
            sq_norm = 0.0
            max_abs = 0.0
            for parameter in model.parameters():
                value = parameter.detach().float()
                checksum += float(value.sum().item())
                sq_norm += float(value.square().sum().item())
                max_abs = max(max_abs, float(value.abs().max().item()))
        del model
        del optimizer
        torch.cuda.empty_cache()
        return {
            "losses": losses,
            "grad_norms": grad_norms,
            "elapsed_sec": elapsed_sec,
            "tokens_per_sec": float(steps * micro_bsz * ctx_len / elapsed_sec),
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_gib": float(peak_memory_bytes / (1024**3)),
            "param_checksum": checksum,
            "param_l2_norm": sq_norm ** 0.5,
            "param_max_abs": max_abs,
            "snapshots": snapshots,
        }

    full = run_variant(0)
    fused = run_variant(fused_chunk)

    loss_diffs = [abs(a - b) for a, b in zip(full["losses"], fused["losses"])]
    grad_norm_diffs = [abs(a - b) for a, b in zip(full["grad_norms"], fused["grad_norms"])]
    param_diffs = {
        name: float((full["snapshots"][name] - fused["snapshots"][name]).abs().max().item())
        for name in full["snapshots"].keys() & fused["snapshots"].keys()
    }
    max_loss_diff = max(loss_diffs) if loss_diffs else 0.0
    max_grad_norm_diff = max(grad_norm_diffs) if grad_norm_diffs else 0.0
    max_param_diff = max(param_diffs.values()) if param_diffs else 0.0
    checksum_diff = abs(full["param_checksum"] - fused["param_checksum"])
    l2_norm_diff = abs(full["param_l2_norm"] - fused["param_l2_norm"])
    loss_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_LOSS_ATOL", "2e-2"))
    loss_rtol = float(os.environ.get("RWKV_SFT_FUSED_CE_LOSS_RTOL", "2e-3"))
    param_atol = float(os.environ.get("RWKV_SFT_FUSED_CE_PARAM_ATOL", "2e-2"))
    param_rtol = float(os.environ.get("RWKV_SFT_FUSED_CE_PARAM_RTOL", "1e-2"))
    loss_allowed = max(
        loss_atol + loss_rtol * max(abs(a), abs(b))
        for a, b in zip(full["losses"], fused["losses"])
    )
    param_allowed = param_atol + param_rtol * max(full["param_max_abs"], fused["param_max_abs"])
    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "steps": steps,
        "micro_bsz": micro_bsz,
        "fused_chunk": fused_chunk,
        "lr": lr,
        "full_logits": {key: value for key, value in full.items() if key != "snapshots"},
        "fused_masked_head": {key: value for key, value in fused.items() if key != "snapshots"},
        "loss_diffs": loss_diffs,
        "max_loss_diff": max_loss_diff,
        "loss_allowed": loss_allowed,
        "grad_norm_diffs": grad_norm_diffs,
        "max_grad_norm_diff": max_grad_norm_diff,
        "param_snapshot_max_abs_diffs": param_diffs,
        "max_param_snapshot_diff": max_param_diff,
        "param_allowed": param_allowed,
        "param_checksum_diff": checksum_diff,
        "param_l2_norm_diff": l2_norm_diff,
        "memory_delta_gib": full["peak_memory_gib"] - fused["peak_memory_gib"],
        "fused_peak_memory_ratio": fused["peak_memory_bytes"] / max(full["peak_memory_bytes"], 1),
        "fused_elapsed_ratio": fused["elapsed_sec"] / max(full["elapsed_sec"], 1e-9),
    }
    summary_file = os.environ.get("RWKV_SFT_FUSED_CE_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert max_loss_diff <= loss_allowed, json.dumps(summary, indent=2)
    assert max_param_diff <= param_allowed, json.dumps(summary, indent=2)
    if os.environ.get("RWKV_SFT_FUSED_CE_REQUIRE_MEMORY_IMPROVEMENT", "") == "1":
        assert fused["peak_memory_bytes"] < full["peak_memory_bytes"], json.dumps(summary, indent=2)


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

    train_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=1,
        epoch_count=1,
    )
    _run_train_py(command, "train.py SFT CUDA smoke")

    assert (proj_dir / "train_log.txt").is_file()


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_accumulation_loss_matches_large_micro_batch(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_DP_ZERO_ACCUM_EQUIV_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_DP_ZERO_ACCUM_EQUIV_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DP/ZeRO accumulation equivalence smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DP/ZeRO accumulation equivalence smoke requires a DeepSpeed strategy")

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    prefix = _build_accum_equiv_sft_binidx(tmp_path, pad_length, docs=devices * 2, vocab_size=dims["vocab_size"])
    large_proj_dir = tmp_path / "ds_large_micro_bsz"
    accum_proj_dir = tmp_path / "ds_accum"
    fused_chunk = int(
        os.environ.get(
            "RWKV_SFT_DP_ZERO_ACCUM_EQUIV_FUSED_CHUNK",
            os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512"),
        )
    )

    common_extra_args = [
        "--epoch_save",
        "0",
        "--keep_last_n_checkpoints",
        "0",
    ]
    large_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=large_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=1,
        epoch_count=1,
        micro_bsz=2,
        accumulate_grad_batches=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args,
    )
    accum_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=accum_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=1,
        epoch_count=1,
        micro_bsz=1,
        accumulate_grad_batches=2,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args,
    )

    _run_train_py(large_command, "train.py SFT DeepSpeed large micro-batch equivalence run")
    _run_train_py(accum_command, "train.py SFT DeepSpeed accumulated micro-batch equivalence run")

    large_loss = _read_train_log_epoch_loss(large_proj_dir)
    accum_loss = _read_train_log_epoch_loss(accum_proj_dir)
    diff = abs(large_loss - accum_loss)
    atol = float(os.environ.get("RWKV_SFT_DP_ZERO_ACCUM_EQUIV_ATOL", "1e-2"))
    rtol = float(os.environ.get("RWKV_SFT_DP_ZERO_ACCUM_EQUIV_RTOL", "1e-3"))
    allowed = atol + rtol * abs(large_loss)
    summary = {
        "devices": devices,
        "strategy": strategy,
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "fused_chunk": fused_chunk,
        "large_micro_bsz_loss": large_loss,
        "accumulated_loss": accum_loss,
        "loss_diff": diff,
        "allowed_diff": allowed,
        "large_proj_dir": str(large_proj_dir),
        "accum_proj_dir": str(accum_proj_dir),
    }
    summary_file = os.environ.get("RWKV_SFT_DP_ZERO_ACCUM_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert diff <= allowed, json.dumps(summary, indent=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_masked_fused_ce_smoke(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_FUSED_CE_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DeepSpeed fused CE smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DeepSpeed fused CE smoke requires a DeepSpeed strategy")

    epoch_steps = int(os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_STEPS", "2"))
    fused_chunk = int(os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512"))
    assert epoch_steps >= 1
    assert fused_chunk > 0

    prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    proj_dir = tmp_path / "fused_ce_train_py"

    train_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
    )
    started_at = time.perf_counter()
    output = _run_train_py(command, "train.py SFT DeepSpeed masked fused CE")
    elapsed_sec = time.perf_counter() - started_at
    loss = _read_train_log_epoch_loss(proj_dir)

    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "epoch_steps": epoch_steps,
        "fused_chunk": fused_chunk,
        "loss": loss,
        "elapsed_sec": elapsed_sec,
        "proj_dir": str(proj_dir),
        "output_tail": output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert torch.isfinite(torch.tensor(loss)).item(), json.dumps(summary, indent=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_masked_fused_ce_gather_verify_smoke(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_FUSED_CE_GATHER_VERIFY_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_FUSED_CE_GATHER_VERIFY_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DeepSpeed gather-verify smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DeepSpeed gather-verify smoke requires a DeepSpeed strategy")

    epoch_steps = int(os.environ.get("RWKV_SFT_FUSED_CE_GATHER_VERIFY_STEPS", "2"))
    fused_chunk = int(os.environ.get("RWKV_SFT_FUSED_CE_GATHER_VERIFY_CHUNK", os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512")))
    gather_tol = float(os.environ.get("RWKV_SFT_GATHER_VERIFY_TOL", "5e-4"))
    assert epoch_steps >= 1
    assert fused_chunk > 0

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    docs = int(os.environ.get("RWKV_SFT_FUSED_CE_GATHER_VERIFY_DOCS", str(max(devices * epoch_steps, 16))))
    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length,
        docs=docs,
        vocab_size=dims["vocab_size"],
        name="gather_verify_sft",
    )
    proj_dir = tmp_path / "fused_ce_gather_verify_train_py"
    wrapper_path = _write_train_py_sft_gather_verify_wrapper(tmp_path / "train_py_sft_gather_verify_wrapper.py")

    train_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        accumulate_grad_batches=1,
    )
    master_port = int(os.environ.get("RWKV_SFT_FUSED_CE_GATHER_VERIFY_MASTER_PORT", "29617"))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(devices),
        "--master_port",
        str(master_port),
        str(wrapper_path),
        *train_command[2:],
    ]
    env = os.environ.copy()
    env["RWKV_SFT_GATHER_VERIFY_TOL"] = str(gather_tol)
    verify_log = tmp_path / "gather_verify.jsonl"
    env["RWKV_SFT_GATHER_VERIFY_LOG"] = str(verify_log)
    output = _run_train_py(command, "train.py SFT DeepSpeed fused CE gather verify", env=env)
    loss = _read_train_log_epoch_loss(proj_dir)

    verify_events = []
    if verify_log.is_file():
        verify_events = [
            json.loads(line)
            for line in verify_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    verified_events = [event for event in verify_events if event.get("event") == "verified"]
    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "epoch_steps": epoch_steps,
        "docs": docs,
        "fused_chunk": fused_chunk,
        "gather_tol": gather_tol,
        "master_port": master_port,
        "verify_event_count": len(verified_events),
        "verify_events": verify_events,
        "loss": loss,
        "proj_dir": str(proj_dir),
        "output_tail": output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_FUSED_CE_GATHER_VERIFY_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert len(verified_events) >= epoch_steps, json.dumps(summary, indent=2)
    assert torch.isfinite(torch.tensor(loss)).item(), json.dumps(summary, indent=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_tail_eval_heldout_smoke(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_TAIL_EVAL_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_TAIL_EVAL_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DeepSpeed tail eval smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DeepSpeed tail eval smoke requires a DeepSpeed strategy")

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length,
        docs=max(devices * 2, 4),
        vocab_size=dims["vocab_size"],
        name="tail_eval_heldout_sft",
    )
    proj_dir = tmp_path / "tail_eval_heldout_train_py"
    fused_chunk = int(os.environ.get("RWKV_SFT_TAIL_EVAL_FUSED_CHUNK", os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512")))

    command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=2,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=[
            "--save_every_n_steps",
            "1",
            "--keep_last_n_checkpoints",
            "2",
            "--sft_eval_tail_docs",
            "1",
            "--sft_eval_every_n_steps",
            "1",
            "--sft_eval_steps",
            "1",
        ],
    )
    output = _run_train_py(command, "train.py SFT DeepSpeed tail eval")
    log_text = (proj_dir / "train_log.txt").read_text(encoding="utf-8")
    assert "eval step 1" in log_text
    assert "mode heldout" in output
    _assert_step_checkpoint_exists(proj_dir)
    assert torch.isfinite(torch.tensor(_read_train_log_epoch_loss(proj_dir))).item()

    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "fused_chunk": fused_chunk,
        "eval_mode": "heldout",
        "proj_dir": str(proj_dir),
        "output_tail": output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_TAIL_EVAL_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_tail_eval_overlap_smoke(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_TAIL_EVAL_OVERLAP_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_TAIL_EVAL_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DeepSpeed tail eval overlap smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DeepSpeed tail eval overlap smoke requires a DeepSpeed strategy")

    prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    proj_dir = tmp_path / "tail_eval_overlap_train_py"
    fused_chunk = int(os.environ.get("RWKV_SFT_TAIL_EVAL_FUSED_CHUNK", os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512")))

    command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=2,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=[
            "--save_every_n_steps",
            "1",
            "--keep_last_n_checkpoints",
            "2",
            "--sft_eval_tail_docs",
            "1",
            "--sft_eval_include_in_train",
            "1",
            "--sft_eval_every_n_steps",
            "1",
            "--sft_eval_steps",
            "1",
        ],
    )
    output = _run_train_py(command, "train.py SFT DeepSpeed tail eval overlap")
    log_text = (proj_dir / "train_log.txt").read_text(encoding="utf-8")
    assert "eval step 1" in log_text
    assert "mode overlap" in output
    _assert_step_checkpoint_exists(proj_dir)


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_shuffle_tail_eval_smoke(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_SHUFFLE_TAIL_EVAL_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SHUFFLE_TAIL_EVAL_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DeepSpeed shuffle tail eval smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DeepSpeed shuffle tail eval smoke requires a DeepSpeed strategy")

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    docs = int(os.environ.get("RWKV_SFT_SHUFFLE_TAIL_EVAL_DOCS", str(max(devices * 4, 16))))
    eval_docs = int(os.environ.get("RWKV_SFT_SHUFFLE_TAIL_EVAL_HELDOUT_DOCS", str(max(2, devices // 2))))
    shuffle_seed = int(os.environ.get("RWKV_SFT_SHUFFLE_TAIL_EVAL_SEED", "31415"))
    assert docs > eval_docs > 0

    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length,
        docs=docs,
        vocab_size=dims["vocab_size"],
        name="shuffle_tail_eval_sft",
    )
    split_train, split_eval = compute_sft_shuffled_split_indices(docs, eval_docs, seed=shuffle_seed)
    assert set(split_train.tolist()).isdisjoint(set(split_eval.tolist()))
    assert split_eval.tolist() != list(range(docs - eval_docs, docs))

    proj_dir = tmp_path / "shuffle_tail_eval_train_py"
    fused_chunk = int(os.environ.get("RWKV_SFT_SHUFFLE_TAIL_EVAL_FUSED_CHUNK", os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512")))
    epoch_steps = int(os.environ.get("RWKV_SFT_SHUFFLE_TAIL_EVAL_STEPS", "2"))

    command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=[
            "--sft_train_shuffle",
            "1",
            "--sft_train_shuffle_seed",
            str(shuffle_seed),
            "--sft_eval_tail_docs",
            str(eval_docs),
            "--sft_eval_every_n_steps",
            "1",
            "--sft_eval_steps",
            "1",
            "--save_every_n_steps",
            "1",
            "--keep_last_n_checkpoints",
            "2",
        ],
    )
    output = _run_train_py(command, "train.py SFT DeepSpeed shuffle tail eval")
    log_text = (proj_dir / "train_log.txt").read_text(encoding="utf-8")
    assert "eval step 1" in log_text
    assert "mode heldout" in output
    assert "train_shuffle=1" in output
    _assert_step_checkpoint_exists(proj_dir)
    loss = _read_train_log_epoch_loss(proj_dir)
    assert torch.isfinite(torch.tensor(loss)).item()

    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "docs": docs,
        "eval_docs": eval_docs,
        "shuffle_seed": shuffle_seed,
        "split_eval": split_eval.tolist(),
        "raw_tail": list(range(docs - eval_docs, docs)),
        "epoch_steps": epoch_steps,
        "fused_chunk": fused_chunk,
        "loss": loss,
        "proj_dir": str(proj_dir),
        "output_tail": output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_SHUFFLE_TAIL_EVAL_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_shuffle_resume_smoke(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_SHUFFLE_RESUME_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SHUFFLE_RESUME_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DeepSpeed shuffle resume smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DeepSpeed shuffle resume smoke requires a DeepSpeed strategy")

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    docs = int(os.environ.get("RWKV_SFT_SHUFFLE_RESUME_DOCS", str(max(devices * 4, 16))))
    eval_docs = int(os.environ.get("RWKV_SFT_SHUFFLE_RESUME_HELDOUT_DOCS", str(max(2, devices // 2))))
    shuffle_seed = int(os.environ.get("RWKV_SFT_SHUFFLE_RESUME_SEED", "27182"))
    epoch_steps = int(os.environ.get("RWKV_SFT_SHUFFLE_RESUME_EPOCH_STEPS", "4"))
    save_step = int(os.environ.get("RWKV_SFT_SHUFFLE_RESUME_SAVE_STEP", "2"))
    assert docs > eval_docs > 0
    assert 0 < save_step < epoch_steps

    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length,
        docs=docs,
        vocab_size=dims["vocab_size"],
        name="shuffle_resume_sft",
    )
    proj_dir = tmp_path / "shuffle_resume_train_py"
    fused_chunk = int(os.environ.get("RWKV_SFT_SHUFFLE_RESUME_FUSED_CHUNK", os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512")))
    common_extra_args = [
        "--sft_train_shuffle",
        "1",
        "--sft_train_shuffle_seed",
        str(shuffle_seed),
        "--sft_eval_tail_docs",
        str(eval_docs),
        "--sft_eval_every_n_steps",
        "1",
        "--sft_eval_steps",
        "1",
        "--epoch_save",
        "0",
        "--keep_last_n_checkpoints",
        "0",
    ]

    first_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args + ["--save_at_step", str(save_step)],
    )
    first_output = _run_train_py(first_command, "train.py SFT DeepSpeed shuffle resume source")
    step_checkpoint = proj_dir / f"rwkv-step-{save_step}.pth"
    assert step_checkpoint.is_dir(), f"expected DeepSpeed checkpoint directory: {step_checkpoint}"

    resume_command = _train_py_command(
        load_model=step_checkpoint,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args,
    )
    resume_output = _run_train_py(resume_command, "train.py SFT DeepSpeed shuffle resume")
    assert "Preloading resume position" in resume_output
    assert "Resuming trainer state" in resume_output
    assert "Resuming dataloader at step offset" in resume_output
    assert "train_shuffle=1" in first_output + resume_output

    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "docs": docs,
        "eval_docs": eval_docs,
        "shuffle_seed": shuffle_seed,
        "epoch_steps": epoch_steps,
        "save_step": save_step,
        "fused_chunk": fused_chunk,
        "proj_dir": str(proj_dir),
        "resume_output_tail": resume_output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_SHUFFLE_RESUME_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_resume_loss_matches_continuous_smoke(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_RESUME_LOSS_EQUIV_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("DeepSpeed resume loss equivalence smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("DeepSpeed resume loss equivalence smoke requires a DeepSpeed strategy")

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    epoch_steps = int(os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_STEPS", "9"))
    save_step = int(os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_SAVE_STEP", "5"))
    accumulate = int(os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_ACCUMULATE_GRAD_BATCHES", "4"))
    tol = float(os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_TOL", "1e-3"))
    assert 0 < save_step < epoch_steps

    docs = int(os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_DOCS", str(max(devices * accumulate * (epoch_steps + 2), 64))))
    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length,
        docs=docs,
        vocab_size=dims["vocab_size"],
        name="resume_loss_equiv_sft",
    )
    fused_chunk = int(os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_FUSED_CHUNK", os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512")))

    common_extra_args = [
        "--epoch_save",
        "0",
        "--save_every_n_steps",
        "0",
        "--keep_last_n_checkpoints",
        "0",
        "--wandb",
        "fake-sft-resume-loss-equiv",
    ]

    continuous_env_dir = tmp_path / "continuous_wandb"
    continuous_env_dir.mkdir()
    continuous_env, continuous_wandb_log = _fake_wandb_env(continuous_env_dir)
    continuous_proj_dir = tmp_path / "resume_loss_continuous"
    continuous_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=continuous_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        accumulate_grad_batches=accumulate,
        extra_args=common_extra_args + ["--save_at_step", str(save_step)],
    )
    continuous_output = _run_train_py(
        continuous_command,
        "train.py SFT DeepSpeed continuous source for resume loss equivalence",
        env=continuous_env,
    )
    step_checkpoint = continuous_proj_dir / f"rwkv-step-{save_step}.pth"
    assert step_checkpoint.is_dir(), f"expected DeepSpeed checkpoint directory: {step_checkpoint}"

    resume_env_dir = tmp_path / "resume_wandb"
    resume_env_dir.mkdir()
    resume_env, resume_wandb_log = _fake_wandb_env(resume_env_dir)
    resume_proj_dir = tmp_path / "resume_loss_resumed"
    resume_command = _train_py_command(
        load_model=step_checkpoint,
        prefix=prefix,
        proj_dir=resume_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        accumulate_grad_batches=accumulate,
        extra_args=common_extra_args,
    )
    resume_output = _run_train_py(resume_command, "train.py SFT DeepSpeed resume loss equivalence", env=resume_env)

    continuous_losses = _read_fake_wandb_train_losses(continuous_wandb_log)
    resume_losses = _read_fake_wandb_train_losses(resume_wandb_log)
    compared_steps = list(range(save_step + 1, epoch_steps + 1))
    diffs = {
        step: abs(continuous_losses[step] - resume_losses[step])
        for step in compared_steps
        if step in continuous_losses and step in resume_losses
    }
    missing = [step for step in compared_steps if step not in continuous_losses or step not in resume_losses]
    assert not missing, {
        "missing_steps": missing,
        "continuous_steps": sorted(continuous_losses),
        "resume_steps": sorted(resume_losses),
    }
    assert all(diff <= tol for diff in diffs.values()), json.dumps(
        {
            "tol": tol,
            "diffs": diffs,
            "continuous_losses": {step: continuous_losses[step] for step in compared_steps},
            "resume_losses": {step: resume_losses[step] for step in compared_steps},
            "continuous_output_tail": continuous_output[-4000:],
            "resume_output_tail": resume_output[-4000:],
        },
        indent=2,
    )

    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "docs": docs,
        "epoch_steps": epoch_steps,
        "save_step": save_step,
        "accumulate_grad_batches": accumulate,
        "fused_chunk": fused_chunk,
        "tol": tol,
        "diffs": diffs,
        "continuous_losses": {step: continuous_losses[step] for step in compared_steps},
        "resume_losses": {step: resume_losses[step] for step in compared_steps},
        "continuous_proj_dir": str(continuous_proj_dir),
        "resume_proj_dir": str(resume_proj_dir),
        "continuous_output_tail": continuous_output[-4000:],
        "resume_output_tail": resume_output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_RESUME_LOSS_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_shuffle_eval_wandb_loss_matches_baseline(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_SHUFFLE_EVAL_WANDB_EQUIV_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("shuffle eval+wandb equivalence smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("shuffle eval+wandb equivalence smoke requires a DeepSpeed strategy")

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    epoch_steps = int(os.environ.get("RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_STEPS", "2"))
    tol = float(os.environ.get("RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_LOSS_TOL", "5e-4"))
    shuffle_seed = int(os.environ.get("RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_SEED", "16180"))
    fused_chunk = int(
        os.environ.get(
            "RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_FUSED_CHUNK",
            os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512"),
        )
    )
    docs = int(os.environ.get("RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_DOCS", str(max(devices * (epoch_steps + 2), 16))))
    eval_docs = int(os.environ.get("RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_HELDOUT_DOCS", str(max(2, devices // 2))))
    assert docs > eval_docs > 0
    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length,
        docs=docs,
        vocab_size=dims["vocab_size"],
        name="shuffle_eval_wandb_equiv_sft",
    )
    _, split_eval = compute_sft_shuffled_split_indices(docs, eval_docs, seed=shuffle_seed)
    assert split_eval.tolist() != list(range(docs - eval_docs, docs))

    common_extra_args = [
        "--sft_train_shuffle",
        "1",
        "--sft_train_shuffle_seed",
        str(shuffle_seed),
        "--sft_eval_tail_docs",
        str(eval_docs),
        "--sft_eval_include_in_train",
        "1",
        "--epoch_save",
        "0",
        "--save_every_n_steps",
        "0",
        "--keep_last_n_checkpoints",
        "0",
    ]
    baseline_proj_dir = tmp_path / "shuffle_eval_wandb_baseline"
    eval_wandb_proj_dir = tmp_path / "shuffle_eval_wandb_enabled"
    baseline_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=baseline_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args,
    )
    eval_wandb_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=eval_wandb_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args
        + [
            "--wandb",
            "fake-sft-shuffle-eval-wandb-equiv",
            "--sft_eval_every_n_steps",
            "1",
            "--sft_eval_steps",
            "1",
        ],
    )

    fake_env, fake_wandb_log = _fake_wandb_env(tmp_path)
    _run_train_py(baseline_command, "train.py SFT shuffle baseline loss equivalence")
    eval_output = _run_train_py(eval_wandb_command, "train.py SFT shuffle eval+wandb loss equivalence", env=fake_env)

    baseline_loss = _read_train_log_epoch_loss(baseline_proj_dir)
    eval_wandb_loss = _read_train_log_epoch_loss(eval_wandb_proj_dir)
    assert abs(baseline_loss - eval_wandb_loss) <= tol, {
        "baseline_loss": baseline_loss,
        "eval_wandb_loss": eval_wandb_loss,
        "tol": tol,
    }

    wandb_events = [
        json.loads(line)
        for line in fake_wandb_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any("train/loss" in event.get("values", {}) for event in wandb_events if event["event"] == "log")
    assert any("eval/loss" in event.get("values", {}) for event in wandb_events if event["event"] == "log")
    assert "eval step 1" in (eval_wandb_proj_dir / "train_log.txt").read_text(encoding="utf-8")
    assert "train_shuffle=1" in eval_output

    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "docs": docs,
        "eval_docs": eval_docs,
        "shuffle_seed": shuffle_seed,
        "fused_chunk": fused_chunk,
        "split_eval": split_eval.tolist(),
        "raw_tail": list(range(docs - eval_docs, docs)),
        "epoch_steps": epoch_steps,
        "baseline_loss": baseline_loss,
        "eval_wandb_loss": eval_wandb_loss,
        "loss_abs_diff": abs(baseline_loss - eval_wandb_loss),
        "tol": tol,
        "output_tail": eval_output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_eval_wandb_loss_matches_baseline(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_EVAL_WANDB_EQUIV_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_EVAL_WANDB_EQUIV_PAD_LENGTH", os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257")))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("eval+wandb equivalence smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("eval+wandb equivalence smoke requires a DeepSpeed strategy")

    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    epoch_steps = int(os.environ.get("RWKV_SFT_EVAL_WANDB_EQUIV_STEPS", "2"))
    tol = float(os.environ.get("RWKV_SFT_EVAL_WANDB_EQUIV_LOSS_TOL", "5e-4"))
    fused_chunk = int(
        os.environ.get(
            "RWKV_SFT_EVAL_WANDB_EQUIV_FUSED_CHUNK",
            os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512"),
        )
    )
    prefix = _build_synthetic_sft_binidx(
        tmp_path,
        pad_length,
        docs=max(devices * (epoch_steps + 1), 8),
        vocab_size=dims["vocab_size"],
        name="eval_wandb_equiv_sft",
    )

    common_extra_args = [
        "--epoch_save",
        "0",
        "--save_every_n_steps",
        "0",
        "--keep_last_n_checkpoints",
        "0",
    ]
    baseline_proj_dir = tmp_path / "eval_wandb_baseline"
    eval_wandb_proj_dir = tmp_path / "eval_wandb_enabled"

    baseline_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=baseline_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args,
    )
    eval_wandb_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=eval_wandb_proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args
        + [
            "--wandb",
            "fake-sft-eval-wandb-equiv",
            "--sft_eval_tail_docs",
            "1",
            "--sft_eval_include_in_train",
            "1",
            "--sft_eval_every_n_steps",
            "1",
            "--sft_eval_steps",
            "1",
        ],
    )

    fake_env, fake_wandb_log = _fake_wandb_env(tmp_path)
    _run_train_py(baseline_command, "train.py SFT baseline loss equivalence")
    eval_output = _run_train_py(eval_wandb_command, "train.py SFT eval+wandb loss equivalence", env=fake_env)

    baseline_loss = _read_train_log_epoch_loss(baseline_proj_dir)
    eval_wandb_loss = _read_train_log_epoch_loss(eval_wandb_proj_dir)
    assert abs(baseline_loss - eval_wandb_loss) <= tol, {
        "baseline_loss": baseline_loss,
        "eval_wandb_loss": eval_wandb_loss,
        "tol": tol,
    }

    wandb_events = [
        json.loads(line)
        for line in fake_wandb_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["event"] == "init" for event in wandb_events)
    assert any("train/loss" in event.get("values", {}) for event in wandb_events if event["event"] == "log")
    assert any("eval/loss" in event.get("values", {}) for event in wandb_events if event["event"] == "log")
    assert "eval step 1" in (eval_wandb_proj_dir / "train_log.txt").read_text(encoding="utf-8")

    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "devices": devices,
        "strategy": strategy,
        "epoch_steps": epoch_steps,
        "fused_chunk": fused_chunk,
        "baseline_loss": baseline_loss,
        "eval_wandb_loss": eval_wandb_loss,
        "loss_abs_diff": abs(baseline_loss - eval_wandb_loss),
        "tol": tol,
        "output_tail": eval_output[-4000:],
    }
    summary_file = os.environ.get("RWKV_SFT_EVAL_WANDB_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_cuda_resume_from_step_checkpoint(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257"))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    proj_dir = tmp_path / "resume_out"

    first_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=2,
        epoch_count=1,
        extra_args=["--save_at_step", "1"],
    )
    _run_train_py(first_command, "train.py SFT CUDA initial run")

    step_checkpoint = proj_dir / "rwkv-step-1.pth"
    assert step_checkpoint.exists()

    resume_command = _train_py_command(
        load_model=step_checkpoint,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=2,
        epoch_count=1,
    )
    resume_output = _run_train_py(resume_command, "train.py SFT CUDA resume run")

    assert "Preloading resume position" in resume_output
    assert "Resuming trainer state" in resume_output


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_resume_keeps_wsd_lr_position(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_WSD_RESUME_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257"))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    devices = int(os.environ.get("RWKV_SFT_SMOKE_DEVICES", "2"))
    strategy = os.environ.get("RWKV_SFT_SMOKE_STRATEGY", "deepspeed_stage_3_offload")
    if devices < 2:
        pytest.skip("WSD resume smoke requires RWKV_SFT_SMOKE_DEVICES >= 2")
    if torch.cuda.device_count() < devices:
        pytest.skip(f"only {torch.cuda.device_count()} CUDA device(s) visible, need {devices}")
    if "deepspeed" not in strategy:
        pytest.skip("WSD resume smoke requires a DeepSpeed strategy")

    prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    proj_dir = tmp_path / "wsd_resume_out"
    lr_init = os.environ.get("RWKV_SFT_WSD_RESUME_LR_INIT", "1e-4")
    lr_final = os.environ.get("RWKV_SFT_WSD_RESUME_LR_FINAL", "1e-5")
    epoch_steps = int(os.environ.get("RWKV_SFT_WSD_RESUME_EPOCH_STEPS", "33"))
    decay_iters = int(os.environ.get("RWKV_SFT_WSD_RESUME_DECAY_ITERS", "9"))
    warmup_steps = int(os.environ.get("RWKV_SFT_WSD_RESUME_WARMUP_STEPS", "4"))
    fused_chunk = int(
        os.environ.get(
            "RWKV_SFT_WSD_RESUME_FUSED_CHUNK",
            os.environ.get("RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK", "512"),
        )
    )
    decay_start = epoch_steps - decay_iters
    save_step = int(os.environ.get("RWKV_SFT_WSD_RESUME_SAVE_STEP", str(decay_start + (decay_iters - 1) // 2)))
    assert epoch_steps > decay_iters > 0
    assert 0 <= warmup_steps < save_step < epoch_steps
    assert decay_iters % 2 == 1, "this smoke uses an odd decay_iters value so the midpoint is exact"
    assert 2 * (save_step - decay_start) == decay_iters - 1, "save_step should be the WSD half-decay point"

    common_extra_args = [
        "--lr_init",
        lr_init,
        "--lr_final",
        lr_final,
        "--lr_wsd_decay_iters",
        str(decay_iters),
        "--lr_wsd_decay_style",
        "linear",
        "--warmup_steps",
        str(warmup_steps),
        "--save_at_step",
        str(save_step),
        "--epoch_save",
        "0",
        "--keep_last_n_checkpoints",
        "0",
    ]
    first_command = _train_py_command(
        load_model=model_path,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=common_extra_args,
    )
    _run_train_py(first_command, "train.py SFT DeepSpeed WSD initial run")

    step_checkpoint = proj_dir / f"rwkv-step-{save_step}.pth"
    assert step_checkpoint.is_dir(), f"expected DeepSpeed checkpoint directory: {step_checkpoint}"

    resume_command = _train_py_command(
        load_model=step_checkpoint,
        prefix=prefix,
        proj_dir=proj_dir,
        dims=dims,
        ctx_len=ctx_len,
        epoch_steps=epoch_steps,
        epoch_count=1,
        devices=devices,
        strategy=strategy,
        sft_masked_fused_ce_chunk=fused_chunk,
        extra_args=[
            "--lr_init",
            lr_init,
            "--lr_final",
            lr_final,
            "--lr_wsd_decay_iters",
            str(decay_iters),
            "--lr_wsd_decay_style",
            "linear",
            "--warmup_steps",
            str(warmup_steps),
            "--epoch_save",
            "0",
            "--keep_last_n_checkpoints",
            "0",
        ],
    )
    resume_output = _run_train_py(resume_command, "train.py SFT DeepSpeed WSD resume run")
    assert "Preloading resume position" in resume_output
    assert "Resuming trainer state" in resume_output

    resumed_lr = _read_train_log_last_lr(proj_dir)
    summary = {
        "devices": devices,
        "strategy": strategy,
        "fused_chunk": fused_chunk,
        "epoch_steps": epoch_steps,
        "warmup_steps": warmup_steps,
        "lr_wsd_decay_iters": decay_iters,
        "decay_start": decay_start,
        "save_step": save_step,
        "lr_init": float(lr_init),
        "lr_final": float(lr_final),
        "expected_lr_at_save_step": float(lr_init) + (float(lr_final) - float(lr_init)) * 0.5,
        "resumed_lr": resumed_lr,
        "proj_dir": str(proj_dir),
    }
    summary_file = os.environ.get("RWKV_SFT_WSD_RESUME_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert resumed_lr == pytest.approx(float(lr_final), abs=5e-8), json.dumps(summary, indent=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_train_py_sft_deepspeed_checkpoint_converts_to_pth(tmp_path):
    if os.environ.get("RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE") != "1":
        pytest.skip("set RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1 to run this SFT checkpoint merge smoke test")

    checkpoint_dir_env = os.environ.get("RWKV_SFT_MERGE_CHECKPOINT_DIR", "")
    if checkpoint_dir_env:
        checkpoint_dir = Path(checkpoint_dir_env).expanduser()
        if not checkpoint_dir.is_dir():
            pytest.skip(f"RWKV_SFT_MERGE_CHECKPOINT_DIR is not a directory: {checkpoint_dir}")
    else:
        model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE")
        pad_length = int(os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257"))
        ctx_len = pad_length - 1
        assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

        prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
        state = _load_state_dict(model_path)
        dims = _infer_rwkv7_dims(state)
        proj_dir = tmp_path / "merge_out"

        train_command = _train_py_command(
            load_model=model_path,
            prefix=prefix,
            proj_dir=proj_dir,
            dims=dims,
            ctx_len=ctx_len,
            epoch_steps=2,
            epoch_count=1,
            extra_args=["--save_at_step", "1"],
        )
        _run_train_py(train_command, "train.py SFT DeepSpeed merge source run")
        checkpoint_dir = proj_dir / "rwkv-step-1.pth"

    assert checkpoint_dir.is_dir(), (
        "checkpoint merge smoke requires a DeepSpeed sharded checkpoint directory. "
        "Set RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_2, deepspeed_stage_3, or deepspeed_stage_3_offload."
    )

    dtype = os.environ.get("RWKV_SFT_MERGE_DTYPE", "bf16")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    assert dtype in dtype_map

    output_env = os.environ.get("RWKV_SFT_MERGE_OUTPUT_FILE", "")
    output_file = Path(output_env).expanduser() if output_env else tmp_path / f"{checkpoint_dir.stem}.{dtype}.pth"
    if output_file.exists() and os.environ.get("RWKV_SFT_MERGE_OVERWRITE", "") != "1":
        pytest.fail(f"merge output already exists, set RWKV_SFT_MERGE_OVERWRITE=1 to overwrite: {output_file}")

    summary_env = os.environ.get("RWKV_SFT_MERGE_SUMMARY_FILE", "")
    summary_file = Path(summary_env).expanduser() if summary_env else output_file.with_suffix(".summary.txt")
    equivalence_env = os.environ.get("RWKV_SFT_MERGE_EQUIV_FILE", "")
    equivalence_file = Path(equivalence_env).expanduser() if equivalence_env else output_file.with_suffix(".equivalence.json")

    convert_command = [
        sys.executable,
        str(ROOT / "scripts" / "convert_deepspeed_checkpoint_to_pth.py"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-file",
        str(output_file),
        "--dtype",
        dtype,
        "--summary-file",
        str(summary_file),
    ]
    verify_summary = os.environ.get("RWKV_SFT_MERGE_VERIFY_SUMMARY_FILE", "")
    if verify_summary:
        convert_command.extend(["--verify-summary-file", verify_summary])
    if os.environ.get("RWKV_SFT_MERGE_NO_LAZY", "") == "1":
        convert_command.append("--no-lazy-mode")

    _run_checkpoint_merge_command(convert_command, "DeepSpeed checkpoint to pth conversion")
    assert output_file.is_file()
    assert summary_file.is_file()

    loaded = torch.load(output_file, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict)
    assert "emb.weight" in loaded
    assert loaded["emb.weight"].dtype == dtype_map[dtype]

    equivalence_command = [
        sys.executable,
        str(ROOT / "scripts" / "test_converted_checkpoint_equivalence.py"),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--converted-file",
        str(output_file),
        "--dtype",
        dtype,
        "--summary-file",
        str(equivalence_file),
    ]
    merge_device = os.environ.get("RWKV_SFT_MERGE_DEVICE", "")
    if merge_device:
        equivalence_command.extend(["--device", merge_device])
    merge_prompt = os.environ.get("RWKV_SFT_MERGE_PROMPT", "")
    if merge_prompt:
        equivalence_command.extend(["--prompt", merge_prompt])
    if os.environ.get("RWKV_SFT_MERGE_STRICT_FORWARD", "") == "1":
        equivalence_command.append("--strict-forward")
    if os.environ.get("RWKV_SFT_MERGE_NO_LAZY", "") == "1":
        equivalence_command.append("--no-lazy-mode")

    equivalence_output = _run_checkpoint_merge_command(equivalence_command, "converted pth equivalence check")
    assert "[equiv] PASS" in equivalence_output
    assert equivalence_file.is_file()

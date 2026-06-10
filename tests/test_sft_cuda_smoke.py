import json
import os
import subprocess
import sys
import time
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
        encoded_docs.append(
            EncodedDocument(
                input_ids=input_ids,
                loss_mask=[0] + [1] * (pad_length - 1),
            )
        )
    write_documents(str(prefix), encoded_docs)
    return prefix


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
    accumulate_grad_batches: int | None = None,
    devices: int | str | None = None,
    strategy: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    if accumulate_grad_batches is None:
        accumulate_grad_batches = int(os.environ.get("RWKV_SFT_SMOKE_ACCUMULATE_GRAD_BATCHES", "1"))
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


def _run_train_py(command: list[str], label: str) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("RWKV_SFT_SMOKE_TIMEOUT", "1800")),
    )
    combined_output = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        pytest.fail(f"{label} failed with code {result.returncode}\n{combined_output[-8000:]}")
    return combined_output


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
        sft_masked_ce_chunk=int(os.environ.get("RWKV_SFT_SMOKE_MASKED_CE_CHUNK", "0")),
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
    )
    model = RWKV(model_args).to(device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = MyDataset(_base_sft_args(prefix, dims, ctx_len, epoch_steps=accum_steps))
    examples = [dataset[idx] for idx in range(accum_steps)]
    large_batch = tuple(torch.stack([example[field] for example in examples]).cuda(non_blocking=True) for field in range(3))

    with torch.no_grad():
        large_batch_loss = model.training_step(large_batch, 0).detach().float()
        accumulated_loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        weighted_loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        mask_count_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        mask_counts = []
        micro_losses = []

        for example in examples:
            micro_batch = tuple(t.unsqueeze(0).cuda(non_blocking=True) for t in example)
            micro_loss = model.training_step(micro_batch, 0).detach().float()
            mask_count = micro_batch[2].sum().detach().float()
            assert mask_count.item() > 0
            accumulated_loss_sum += micro_loss
            weighted_loss_sum += micro_loss * mask_count
            mask_count_sum += mask_count
            mask_counts.append(mask_count.item())
            micro_losses.append(micro_loss.item())

        accumulated_loss = accumulated_loss_sum / accum_steps
        weighted_accumulated_loss = weighted_loss_sum / mask_count_sum

    loss_value = large_batch_loss.item()
    accumulated_value = accumulated_loss.item()
    weighted_value = weighted_accumulated_loss.item()
    accumulated_diff = abs(loss_value - accumulated_value)
    weighted_diff = abs(loss_value - weighted_value)
    atol = float(os.environ.get("RWKV_SFT_ACCUM_EQUIV_ATOL", "1e-2"))
    rtol = float(os.environ.get("RWKV_SFT_ACCUM_EQUIV_RTOL", "1e-3"))
    allowed = atol + rtol * abs(loss_value)
    summary = {
        "pad_length": pad_length,
        "ctx_len": ctx_len,
        "accum_steps": accum_steps,
        "large_batch_loss": loss_value,
        "accumulated_micro_loss": accumulated_value,
        "token_weighted_accumulated_loss": weighted_value,
        "accumulated_diff": accumulated_diff,
        "token_weighted_diff": weighted_diff,
        "allowed_diff": allowed,
        "mask_counts": mask_counts,
        "micro_losses": micro_losses,
    }
    summary_file = os.environ.get("RWKV_SFT_ACCUM_EQUIV_SUMMARY_FILE", "")
    if summary_file:
        Path(summary_file).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert accumulated_diff <= allowed, json.dumps(summary, indent=2)
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
def test_train_py_sft_cuda_one_step(tmp_path):
    model_path = _require_cuda_smoke("RWKV_RUN_TRAIN_PY_SFT_SMOKE")
    pad_length = int(os.environ.get("RWKV_SFT_SMOKE_PAD_LENGTH", "257"))
    ctx_len = pad_length - 1
    assert ctx_len > 0 and ctx_len % 16 == 0, "ctx_len must be positive and divisible by the RWKV7 chunk length 16"

    prefix = _build_tiny_sft_binidx(tmp_path, pad_length)
    state = _load_state_dict(model_path)
    dims = _infer_rwkv7_dims(state)
    proj_dir = tmp_path / "out"

    command = _train_py_command(
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

import os
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train
from src import trainer as trainer_mod

_REGRESSION_SCRIPT_PATH = REPO_ROOT / "scripts" / "regression_resume_deepspeed_checkpoint.py"
_REGRESSION_SCRIPT_SPEC = importlib.util.spec_from_file_location("regression_resume_deepspeed_checkpoint", _REGRESSION_SCRIPT_PATH)
assert _REGRESSION_SCRIPT_SPEC is not None and _REGRESSION_SCRIPT_SPEC.loader is not None
resume_smoke = importlib.util.module_from_spec(_REGRESSION_SCRIPT_SPEC)
_REGRESSION_SCRIPT_SPEC.loader.exec_module(resume_smoke)


def _make_ds_checkpoint_dir(base: Path, name: str, ts: int) -> Path:
    path = base / name
    path.mkdir()
    (path / "latest").write_text("checkpoint", encoding="utf-8")
    (path / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_text("", encoding="utf-8")
    os.utime(path / "latest", (ts, ts))
    os.utime(path / "zero_pp_rank_0_mp_rank_00_model_states.pt", (ts, ts))
    os.utime(path, (ts, ts))
    return path


def _make_file_checkpoint(base: Path, name: str, ts: int) -> Path:
    path = base / name
    path.write_bytes(b"checkpoint")
    os.utime(path, (ts, ts))
    return path


def test_parse_epoch_checkpoint_name_only_accepts_epoch_style_names():
    assert train.parse_epoch_checkpoint_name("rwkv-init.pth") == -1
    assert train.parse_epoch_checkpoint_name("rwkv-7.pth") == 7
    assert train.parse_epoch_checkpoint_name("rwkv-step-7.pth") is None
    assert train.parse_epoch_checkpoint_name("not-a-checkpoint.pth") is None


def test_detects_deepspeed_checkpoint_directories(tmp_path: Path):
    ds_dir = _make_ds_checkpoint_dir(tmp_path, "rwkv-1.pth", 100)
    assert train.is_deepspeed_checkpoint_dir(str(ds_dir))

    missing_marker_dir = tmp_path / "rwkv-2.pth"
    missing_marker_dir.mkdir()
    assert not train.is_deepspeed_checkpoint_dir(str(missing_marker_dir))

    wrong_suffix_dir = tmp_path / "rwkv-3"
    wrong_suffix_dir.mkdir()
    (wrong_suffix_dir / "latest").write_text("checkpoint", encoding="utf-8")
    assert not train.is_deepspeed_checkpoint_dir(str(wrong_suffix_dir))


@pytest.mark.parametrize(
    "strategy",
    [
        "deepspeed_stage_1",
        "deepspeed_stage_2_offload",
        "deepspeed_stage_3",
        "deepspeed_stage_3_offload",
    ],
)
def test_resolve_resume_checkpoint_path_accepts_all_deepspeed_strategies(strategy: str, tmp_path: Path):
    ckpt_dir = _make_ds_checkpoint_dir(tmp_path, "rwkv-11.pth", 100)
    assert train.resolve_resume_checkpoint_path(str(ckpt_dir), strategy) == str(ckpt_dir)


def test_resolve_resume_checkpoint_path_rejects_non_deepspeed_strategy(tmp_path: Path):
    ckpt_dir = _make_ds_checkpoint_dir(tmp_path, "rwkv-11.pth", 100)
    with pytest.raises(ValueError, match="DeepSpeed sharded checkpoint"):
        train.resolve_resume_checkpoint_path(str(ckpt_dir), "ddp")


def test_prune_old_checkpoints_keeps_latest_numbered_entries_and_preserves_non_numbered(tmp_path: Path):
    _make_ds_checkpoint_dir(tmp_path, "rwkv-init.pth", 100)
    _make_ds_checkpoint_dir(tmp_path, "rwkv-1.pth", 110)
    _make_ds_checkpoint_dir(tmp_path, "rwkv-2.pth", 120)
    _make_file_checkpoint(tmp_path, "rwkv-3.pth", 130)
    _make_file_checkpoint(tmp_path, "rwkv-final.pth", 140)

    args = SimpleNamespace(proj_dir=str(tmp_path), keep_last_n_checkpoints=2)
    trainer_mod.prune_old_checkpoints(args)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "rwkv-2.pth",
        "rwkv-3.pth",
        "rwkv-final.pth",
        "rwkv-init.pth",
    ]


@pytest.mark.parametrize(
    "strategy",
    [
        "deepspeed_stage_1",
        "deepspeed_stage_2",
        "deepspeed_stage_2_offload",
        "deepspeed_stage_3",
        "deepspeed_stage_3_offload",
    ],
)
def test_my_save_uses_trainer_save_checkpoint_for_all_deepspeed_strategies(strategy: str):
    calls = []

    class DummyTrainer:
        def save_checkpoint(self, path, weights_only=True):
            calls.append((path, weights_only))

    trainer_mod.my_save(SimpleNamespace(strategy=strategy), DummyTrainer(), {"x": 1}, "dummy.pth")
    assert calls == [("dummy.pth", False)]


def test_my_save_uses_torch_save_for_non_deepspeed(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_torch_save(payload, path):
        calls.append((payload, path))

    monkeypatch.setattr(trainer_mod.torch, "save", fake_torch_save)
    trainer_mod.my_save(SimpleNamespace(strategy="ddp"), object(), {"x": 1}, "dummy.pth")
    assert calls == [({"x": 1}, "dummy.pth")]


def test_save_train_checkpoint_prunes_old_deepspeed_directories_and_waits_for_barriers(tmp_path: Path):
    _make_ds_checkpoint_dir(tmp_path, "rwkv-1.pth", 100)

    barrier_calls = []
    save_calls = []

    class DummyStrategy:
        def barrier(self):
            barrier_calls.append("barrier")

    class DummyTrainer:
        def __init__(self):
            self.strategy = DummyStrategy()
            self.is_global_zero = True

        def save_checkpoint(self, path, weights_only=True):
            save_calls.append((path, weights_only))
            ckpt_dir = Path(path)
            ckpt_dir.mkdir()
            (ckpt_dir / "latest").write_text("checkpoint", encoding="utf-8")
            (ckpt_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_text("", encoding="utf-8")

    args = SimpleNamespace(
        strategy="deepspeed_stage_2",
        proj_dir=str(tmp_path),
        keep_last_n_checkpoints=1,
        data_type="binidx",
    )

    trainer_mod.save_train_checkpoint(args, DummyTrainer(), SimpleNamespace(state_dict=lambda: {"x": 1}), str(tmp_path / "rwkv-2.pth"))

    assert save_calls == [(str(tmp_path / "rwkv-2.pth"), False)]
    assert barrier_calls == ["barrier", "barrier"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rwkv-2.pth"]


def test_resume_smoke_defaults_require_real_restore_and_progress_markers():
    assert resume_smoke.default_require_patterns() == [
        "Resuming trainer state from",
        "Restoring states from the checkpoint path at",
    ]
    assert "Epoch " in resume_smoke.default_progress_patterns()
    assert resume_smoke.line_has_failure_marker("KeyError: missing lr_schedulers")


def test_train_callback_initializes_logging_state_when_resuming_mid_run(tmp_path: Path):
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            strategy="deepspeed_stage_3_offload",
            proj_dir=str(tmp_path),
            wandb="",
            my_timestamp="2026-06-02-17-00-00",
            run_name="resume-test",
            epoch_begin=0,
            epoch_steps=5040,
            warmup_steps=10,
            my_exit_tokens=1000,
            ctx_len=16,
            real_bsz=8,
            lr_init=1e-4,
            lr_final=1e-5,
            weight_decay=0.01,
            magic_prime=0,
            save_every_n_steps=0,
            save_at_step=0,
        )
    )
    callback.log = lambda *args, **kwargs: None

    trainer = SimpleNamespace(
        global_step=50,
        is_global_zero=True,
        strategy=SimpleNamespace(config={"zero_optimization": {"stage": 3}}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.01, "my_lr_scale": 1.0}])],
        my_loss_all=torch.tensor([1.25], dtype=torch.float32),
    )

    callback.on_train_batch_start(trainer, object(), None, 0)
    callback.on_train_batch_end(trainer, object(), None, None, 0)

    assert trainer.my_loss_count == 1
    assert trainer.my_loss_sum == pytest.approx(1.25)
    assert (tmp_path / "train_log.txt").exists()
    trainer.my_log.close()

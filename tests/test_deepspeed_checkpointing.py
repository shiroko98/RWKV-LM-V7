import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train
from src import trainer as trainer_mod


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
    assert calls == [("dummy.pth", True)]


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

    class DummyStrategy:
        def barrier(self):
            barrier_calls.append("barrier")

    class DummyTrainer:
        def __init__(self):
            self.strategy = DummyStrategy()
            self.is_global_zero = True

        def save_checkpoint(self, path, weights_only=True):
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

    assert barrier_calls == ["barrier", "barrier"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rwkv-2.pth"]

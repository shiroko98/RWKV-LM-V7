import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train
from src import dataset as dataset_mod
from src import lr_schedule
from src import trainer as trainer_mod
from src.sft_binidx import EncodedDocument, write_documents
from src.sft_loss import masked_cross_entropy, masked_head_cross_entropy


def make_sft_args(prefix: str, **overrides):
    args = SimpleNamespace(
        vocab_size=65536,
        data_file=prefix,
        data_type="sft_binidx",
        sft_mask_file="",
        sft_pad_token_id=65532,
        epoch_steps=2,
        real_bsz=1,
        train_stage=0,
        ctx_len=5,
        magic_prime=0,
        epoch_begin=0,
        resume_epoch=0,
        resume_step_offset=0,
        micro_bsz=1,
        accumulate_grad_batches=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_sft_dataset_reads_mask_sidecar_shifts_mask_and_pads(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    prefix = str(tmp_path / "sft")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[10, 11, 12, 13], loss_mask=[0, 1, 0, 1]),
            EncodedDocument(input_ids=[20, 21, 22], loss_mask=[0, 0, 1]),
        ],
    )

    dataset = dataset_mod.MyDataset(make_sft_args(prefix))
    assert len(dataset) == 2

    x0, y0, mask0 = dataset[0]
    assert torch.equal(x0, torch.tensor([10, 11, 12, 13, 65532], dtype=torch.long))
    assert torch.equal(y0, torch.tensor([11, 12, 13, 65532, 65532], dtype=torch.long))
    assert torch.equal(mask0, torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32))

    x1, y1, mask1 = dataset[1]
    assert torch.equal(x1, torch.tensor([20, 21, 22, 65532, 65532], dtype=torch.long))
    assert torch.equal(y1, torch.tensor([21, 22, 65532, 65532, 65532], dtype=torch.long))
    assert torch.equal(mask1, torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32))


def test_sft_dataset_uses_rank_and_resume_offset_for_document_selection(tmp_path):
    prefix = str(tmp_path / "sft")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[10, 11], loss_mask=[0, 1]),
            EncodedDocument(input_ids=[20, 21], loss_mask=[0, 1]),
            EncodedDocument(input_ids=[30, 31], loss_mask=[0, 1]),
            EncodedDocument(input_ids=[40, 41], loss_mask=[0, 1]),
        ],
    )
    dataset = dataset_mod.MyDataset(make_sft_args(prefix, ctx_len=3, real_bsz=2))
    dataset.world_size = 2
    dataset.global_rank = 1
    dataset.step_offset = 1

    x, y, mask = dataset[0]

    assert torch.equal(x, torch.tensor([40, 41, 65532], dtype=torch.long))
    assert torch.equal(y, torch.tensor([41, 65532, 65532], dtype=torch.long))
    assert torch.equal(mask, torch.tensor([1, 0, 0], dtype=torch.float32))


def test_sft_dataset_uses_gradient_accumulation_for_length_epoch_and_resume(tmp_path):
    prefix = str(tmp_path / "sft")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in range(0, 80, 10)
        ],
    )

    dataset = dataset_mod.MyDataset(
        make_sft_args(
            prefix,
            ctx_len=3,
            epoch_steps=2,
            real_bsz=1,
            accumulate_grad_batches=3,
        )
    )

    assert len(dataset) == 6
    assert dataset.samples_per_epoch == 6

    dataset.real_epoch = 1
    x_epoch, y_epoch, mask_epoch = dataset[0]
    assert torch.equal(x_epoch, torch.tensor([60, 61, 65532], dtype=torch.long))
    assert torch.equal(y_epoch, torch.tensor([61, 65532, 65532], dtype=torch.long))
    assert torch.equal(mask_epoch, torch.tensor([1, 0, 0], dtype=torch.float32))

    dataset.real_epoch = 0
    dataset.step_offset = 1
    x_resume, y_resume, mask_resume = dataset[0]
    assert torch.equal(x_resume, torch.tensor([30, 31, 65532], dtype=torch.long))
    assert torch.equal(y_resume, torch.tensor([31, 65532, 65532], dtype=torch.long))
    assert torch.equal(mask_resume, torch.tensor([1, 0, 0], dtype=torch.float32))


def test_sft_dataset_rejects_too_long_documents_and_bad_masks(tmp_path):
    too_long_prefix = str(tmp_path / "too_long")
    write_documents(
        too_long_prefix,
        [EncodedDocument(input_ids=[1, 2, 3, 4, 5], loss_mask=[0, 1, 1, 1, 1])],
    )
    too_long_dataset = dataset_mod.MyDataset(make_sft_args(too_long_prefix, ctx_len=3))
    with pytest.raises(ValueError, match="exceeds ctx_len"):
        too_long_dataset[0]

    bad_mask_prefix = str(tmp_path / "bad_mask")
    write_documents(
        bad_mask_prefix,
        [EncodedDocument(input_ids=[1, 2, 3], loss_mask=[0, 2, 1])],
    )
    bad_mask_dataset = dataset_mod.MyDataset(make_sft_args(bad_mask_prefix, ctx_len=3))
    with pytest.raises(ValueError, match="0/1"):
        bad_mask_dataset[0]


def test_dataset_prime_helper_covers_pretrain_schedule_checks():
    assert dataset_mod.is_prime(1) is False
    assert dataset_mod.is_prime(2) is True
    assert dataset_mod.is_prime(4) is False
    assert dataset_mod.is_prime(25) is False
    assert dataset_mod.is_prime(29) is True


def test_sft_dataset_initialization_validation_branches(monkeypatch):
    class DummyIndex:
        _dtype_size = 2

    class DummyMMap:
        def __init__(self, sizes):
            self._bin_buffer = bytes(int(sum(sizes)) * 2)
            self._index = DummyIndex()
            self.sizes = torch.tensor(sizes).numpy()

        def __len__(self):
            return len(self.sizes)

    def args(**overrides):
        base = make_sft_args("dummy", ctx_len=3)
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([]))
    with pytest.raises(ValueError, match="at least one"):
        dataset_mod.MyDataset(args())

    datasets = [DummyMMap([3]), DummyMMap([3, 3])]
    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: datasets.pop(0))
    with pytest.raises(ValueError, match="document count"):
        dataset_mod.MyDataset(args())

    datasets = [DummyMMap([3]), DummyMMap([2])]
    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: datasets.pop(0))
    with pytest.raises(ValueError, match="document sizes"):
        dataset_mod.MyDataset(args())

    datasets = [DummyMMap([3]), DummyMMap([3])]
    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: datasets.pop(0))
    with pytest.raises(ValueError, match="ctx_len"):
        dataset_mod.MyDataset(args(ctx_len=0))

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([3]))
    with pytest.raises(ValueError, match="Unsupported"):
        dataset_mod.MyDataset(args(data_type="utf-8"))


def test_masked_cross_entropy_matches_manual_selected_token_average():
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [1.0, 0.0, 5.0],
            ]
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 1, 2]], dtype=torch.long)
    loss_mask = torch.tensor([[1, 0, 1]], dtype=torch.float32)

    loss = masked_cross_entropy(logits, targets, loss_mask)
    manual_losses = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
    expected = (manual_losses[0] + manual_losses[2]) / 2

    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad[0, 1].abs().sum().item() == pytest.approx(0)


def test_masked_cross_entropy_handles_zero_mask_and_validates_shapes():
    logits = torch.randn(1, 2, 4, requires_grad=True)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    zero_mask = torch.zeros(1, 2)

    loss = masked_cross_entropy(logits, targets, zero_mask)
    assert loss.item() == pytest.approx(0)
    loss.backward()
    assert logits.grad.abs().sum().item() == pytest.approx(0)

    with pytest.raises(ValueError, match="logits"):
        masked_cross_entropy(logits.squeeze(0), targets, zero_mask)
    with pytest.raises(ValueError, match="targets"):
        masked_cross_entropy(logits, targets[:, :1], zero_mask)
    with pytest.raises(ValueError, match="loss_mask"):
        masked_cross_entropy(logits, targets, zero_mask[:, :1])


def test_masked_head_cross_entropy_matches_full_logits_and_masks_gradients():
    torch.manual_seed(123)
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)

    full_hidden = hidden.detach().clone().requires_grad_()
    full_weight = weight.detach().clone().requires_grad_()
    expected = masked_cross_entropy(F.linear(full_hidden, full_weight), targets, loss_mask)
    actual = masked_head_cross_entropy(hidden, weight, targets, loss_mask, chunk_size=1)

    assert actual.item() == pytest.approx(expected.item())
    actual.backward()
    expected.backward()

    assert torch.allclose(hidden.grad, full_hidden.grad, atol=1e-6)
    assert torch.allclose(weight.grad, full_weight.grad, atol=1e-6)
    assert hidden.grad[0, 0].abs().sum().item() == pytest.approx(0)
    assert hidden.grad[0, 2].abs().sum().item() == pytest.approx(0)


def test_masked_head_cross_entropy_accepts_linear_head_module():
    torch.manual_seed(321)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    head = torch.nn.Linear(4, 6, bias=False)
    targets = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    loss_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)

    full_hidden = hidden.detach().clone().requires_grad_()
    full_head = torch.nn.Linear(4, 6, bias=False)
    full_head.load_state_dict(head.state_dict())

    expected = masked_cross_entropy(full_head(full_hidden), targets, loss_mask)
    actual = masked_head_cross_entropy(hidden, head, targets, loss_mask, chunk_size=2)

    assert actual.item() == pytest.approx(expected.item())
    actual.backward()
    expected.backward()
    assert torch.allclose(hidden.grad, full_hidden.grad, atol=1e-6)
    assert torch.allclose(head.weight.grad, full_head.weight.grad, atol=1e-6)


def test_masked_head_cross_entropy_zero_mask_and_validation():
    hidden = torch.randn(1, 2, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    zero_mask = torch.zeros(1, 2)

    loss = masked_head_cross_entropy(hidden, weight, targets, zero_mask, chunk_size=2)
    assert loss.item() == pytest.approx(0)
    loss.backward()
    assert hidden.grad.abs().sum().item() == pytest.approx(0)
    assert weight.grad.abs().sum().item() == pytest.approx(0)

    with pytest.raises(ValueError, match="hidden"):
        masked_head_cross_entropy(hidden.squeeze(0), weight, targets, zero_mask)
    with pytest.raises(ValueError, match="weight"):
        masked_head_cross_entropy(hidden, weight[:, :2], targets, zero_mask)
    with pytest.raises(ValueError, match="targets"):
        masked_head_cross_entropy(hidden, weight, targets[:, :1], zero_mask)
    with pytest.raises(ValueError, match="loss_mask"):
        masked_head_cross_entropy(hidden, weight, targets, zero_mask[:, :1])
    with pytest.raises(ValueError, match="chunk_size"):
        masked_head_cross_entropy(hidden, weight, targets, zero_mask, chunk_size=0)


def test_configure_epoch_schedule_preserves_sft_steps_and_keeps_pretrain_schedule():
    sft_args = SimpleNamespace(data_type="sft_binidx", epoch_steps=7, epoch_count=3, real_bsz=8, sft_one_pass=0)
    train.configure_epoch_schedule(sft_args)
    assert sft_args.epoch_steps == 7
    assert sft_args.epoch_count == 3

    pretrain_args = SimpleNamespace(data_type="binidx", epoch_steps=1, epoch_count=1, real_bsz=8, magic_prime=80640)
    train.configure_epoch_schedule(pretrain_args)
    assert pretrain_args.epoch_steps == 5040
    assert pretrain_args.epoch_count == 2

    with pytest.raises(ValueError, match="epoch_steps"):
        train.configure_epoch_schedule(SimpleNamespace(data_type="sft_binidx", epoch_steps=0, epoch_count=1))
    with pytest.raises(ValueError, match="epoch_count"):
        train.configure_epoch_schedule(SimpleNamespace(data_type="sft_binidx", epoch_steps=1, epoch_count=0))
    with pytest.raises(ValueError, match="Unsupported"):
        train.configure_epoch_schedule(SimpleNamespace(data_type="utf-8"))


def test_sft_one_pass_overrides_steps_and_epoch_count(monkeypatch):
    args = SimpleNamespace(
        data_type="sft_binidx",
        data_file="dummy",
        sft_one_pass=1,
        epoch_steps=0,
        epoch_count=0,
        real_bsz=8,
        effective_bsz=32,
        accumulate_grad_batches=4,
    )

    train.configure_epoch_schedule(args)
    monkeypatch.setattr(train, "count_binidx_documents", lambda prefix: 65)
    train.configure_sft_one_pass(args)
    train.configure_samples_per_epoch(args)
    train.configure_training_limits(args)

    assert args.epoch_steps == 3
    assert args.epoch_count == 1
    assert args.max_epochs == 1
    assert args.samples_per_epoch == 96
    assert args.sft_one_pass_documents == 65


def test_sft_one_pass_disabled_sets_default_metadata():
    args = SimpleNamespace(data_type="sft_binidx", sft_one_pass=0, epoch_steps=7, epoch_count=3)
    train.configure_sft_one_pass(args)
    assert args.sft_one_pass == 0
    assert args.sft_one_pass_documents == 0
    assert args.epoch_steps == 7
    assert args.epoch_count == 3


def test_count_binidx_documents_reads_index_only(tmp_path):
    prefix = str(tmp_path / "count_docs")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[1, 2], loss_mask=[0, 1]),
            EncodedDocument(input_ids=[3, 4], loss_mask=[0, 1]),
            EncodedDocument(input_ids=[5, 6], loss_mask=[0, 1]),
        ],
    )

    assert train.count_binidx_documents(prefix) == 3


def test_sft_one_pass_rejects_unsupported_or_empty_datasets(monkeypatch):
    with pytest.raises(ValueError, match="sft_binidx"):
        train.configure_sft_one_pass(SimpleNamespace(data_type="binidx", sft_one_pass=1, effective_bsz=8))

    with pytest.raises(ValueError, match="effective_bsz"):
        train.configure_sft_one_pass(
            SimpleNamespace(data_type="sft_binidx", data_file="empty", sft_one_pass=1, effective_bsz=0)
        )

    monkeypatch.setattr(train, "count_binidx_documents", lambda prefix: 0)
    with pytest.raises(ValueError, match="at least one document"):
        train.configure_sft_one_pass(
            SimpleNamespace(data_type="sft_binidx", data_file="empty", sft_one_pass=1, effective_bsz=8)
        )


def test_configure_training_limits_stops_sft_by_epoch_count_and_preserves_pretrain_default():
    sft_args = SimpleNamespace(data_type="sft_binidx", epoch_count=3, max_epochs=-1)
    train.configure_training_limits(sft_args)
    assert sft_args.max_epochs == 3

    pretrain_args = SimpleNamespace(data_type="binidx", epoch_count=3, max_epochs=3)
    train.configure_training_limits(pretrain_args)
    assert pretrain_args.max_epochs == -1


def test_batch_size_helpers_track_sft_gradient_accumulation():
    sft_args = SimpleNamespace(
        data_type="sft_binidx",
        epoch_steps=5,
        real_bsz=8,
        accumulate_grad_batches="4",
    )
    train.configure_batch_sizes(sft_args)
    train.configure_samples_per_epoch(sft_args)

    assert sft_args.accumulate_grad_batches == 4
    assert sft_args.effective_bsz == 32
    assert sft_args.samples_per_epoch == 160

    pretrain_args = SimpleNamespace(
        data_type="binidx",
        epoch_steps=5,
        real_bsz=8,
        accumulate_grad_batches=4,
    )
    train.configure_batch_sizes(pretrain_args)
    train.configure_samples_per_epoch(pretrain_args)
    assert pretrain_args.effective_bsz == 8
    assert pretrain_args.samples_per_epoch == 40

    none_accum_args = SimpleNamespace(data_type="sft_binidx", real_bsz=2, accumulate_grad_batches=None)
    train.configure_batch_sizes(none_accum_args)
    assert none_accum_args.accumulate_grad_batches == 1
    assert none_accum_args.effective_bsz == 2

    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        train.normalize_accumulate_grad_batches(SimpleNamespace(accumulate_grad_batches=0))
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        train.normalize_accumulate_grad_batches(SimpleNamespace(accumulate_grad_batches="bad"))


def test_validate_sft_loss_settings_accepts_one_backend_and_rejects_invalid_values():
    train.validate_sft_loss_settings(SimpleNamespace(sft_masked_ce_chunk=0, sft_masked_fused_ce_chunk=0))
    train.validate_sft_loss_settings(SimpleNamespace(sft_masked_ce_chunk=128, sft_masked_fused_ce_chunk=0))
    train.validate_sft_loss_settings(SimpleNamespace(sft_masked_ce_chunk=0, sft_masked_fused_ce_chunk=4096))

    with pytest.raises(ValueError, match="sft_masked_ce_chunk"):
        train.validate_sft_loss_settings(SimpleNamespace(sft_masked_ce_chunk=-1, sft_masked_fused_ce_chunk=0))
    with pytest.raises(ValueError, match="sft_masked_fused_ce_chunk"):
        train.validate_sft_loss_settings(SimpleNamespace(sft_masked_ce_chunk=0, sft_masked_fused_ce_chunk=-1))
    with pytest.raises(ValueError, match="either"):
        train.validate_sft_loss_settings(SimpleNamespace(sft_masked_ce_chunk=128, sft_masked_fused_ce_chunk=4096))


def test_checkpoint_path_helpers_handle_empty_regular_and_unreadable_paths(tmp_path, monkeypatch):
    assert train.resolve_resume_checkpoint_path("", "deepspeed_stage_2") is None
    assert train.resolve_resume_checkpoint_path(str(tmp_path / "plain.pth"), "deepspeed_stage_2") is None

    monkeypatch.setattr(train.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(train.os, "listdir", lambda path: (_ for _ in ()).throw(OSError("denied")))
    assert train.is_deepspeed_checkpoint_dir(str(tmp_path / "unreadable.pth")) is False


def test_train_callback_uses_lr_init_when_exit_tokens_disabled(tmp_path):
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="",
            proj_dir=str(tmp_path),
            wandb="",
            my_timestamp="2026-06-05-12-00-00",
            run_name="sft-lr-test",
            epoch_begin=0,
            epoch_steps=2,
            warmup_steps=-1,
            my_exit_tokens=0,
            ctx_len=16,
            real_bsz=1,
            lr_init=2e-4,
            lr_final=1e-5,
            lr_wsd_decay_iters=0,
            lr_wsd_decay_style="cosine",
            weight_decay=0.0,
        )
    )

    trainer = SimpleNamespace(
        global_step=0,
        is_global_zero=True,
        strategy=SimpleNamespace(config={}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.0, "my_lr_scale": 1.0}])],
    )

    callback.on_train_batch_start(trainer, object(), None, 0)

    assert trainer.my_lr == pytest.approx(2e-4)
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(2e-4)
    trainer.my_log.close()


def test_sft_wsd_lr_schedule_supports_cosine_linear_and_default():
    args = SimpleNamespace(
        data_type="sft_binidx",
        epoch_begin=0,
        epoch_steps=10,
        epoch_count=1,
        lr_init=1e-4,
        lr_final=1e-5,
        lr_wsd_decay_iters=4,
        lr_wsd_decay_style="cosine",
    )

    assert lr_schedule.compute_sft_wsd_lr(args, 0) == pytest.approx(1e-4)
    assert lr_schedule.compute_sft_wsd_lr(args, 5) == pytest.approx(1e-4)
    assert lr_schedule.compute_sft_wsd_lr(args, 6) == pytest.approx(1e-4)
    assert lr_schedule.compute_sft_wsd_lr(args, 9) == pytest.approx(1e-5)

    args.epoch_begin = 3
    assert lr_schedule.compute_sft_wsd_lr(args, 39) == pytest.approx(1e-5)
    args.epoch_begin = 0

    args.lr_wsd_decay_style = "linear"
    assert lr_schedule.compute_sft_wsd_lr(args, 8) == pytest.approx(4e-5)

    args.lr_wsd_decay_style = "none"
    assert lr_schedule.compute_sft_wsd_lr(args, 9) == pytest.approx(1e-4)

    args.data_type = "binidx"
    args.lr_wsd_decay_style = "cosine"
    assert lr_schedule.compute_sft_wsd_lr(args, 9) == pytest.approx(1e-4)


def test_sft_wsd_lr_schedule_validates_inputs():
    args = SimpleNamespace(
        data_type="sft_binidx",
        epoch_begin=0,
        epoch_steps=10,
        epoch_count=1,
        lr_init=1e-4,
        lr_final=1e-5,
        lr_wsd_decay_iters=-1,
        lr_wsd_decay_style="cosine",
    )
    with pytest.raises(ValueError, match="lr_wsd_decay_iters"):
        lr_schedule.compute_sft_wsd_lr(args, 0)

    args.lr_wsd_decay_iters = 2
    args.lr_wsd_decay_style = "bad"
    with pytest.raises(ValueError, match="lr_wsd_decay_style"):
        lr_schedule.compute_sft_wsd_lr(args, 0)

    args.lr_wsd_decay_iters = 2
    args.lr_wsd_decay_style = "cosine"
    args.epoch_steps = 0
    with pytest.raises(ValueError, match="positive epoch_steps"):
        lr_schedule.compute_sft_wsd_lr(args, 0)


def test_train_callback_applies_sft_wsd_lr_with_group_scale_and_warmup(tmp_path):
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="",
            proj_dir=str(tmp_path),
            wandb="",
            my_timestamp="2026-06-09-12-00-00",
            run_name="sft-wsd-lr-test",
            epoch_begin=0,
            epoch_steps=10,
            epoch_count=1,
            warmup_steps=10,
            my_exit_tokens=0,
            ctx_len=16,
            real_bsz=1,
            lr_init=1e-4,
            lr_final=1e-5,
            lr_wsd_decay_iters=10,
            lr_wsd_decay_style="linear",
            weight_decay=0.01,
        )
    )

    trainer = SimpleNamespace(
        global_step=5,
        is_global_zero=True,
        strategy=SimpleNamespace(config={}),
        optimizers=[
            SimpleNamespace(
                param_groups=[
                    {"weight_decay": 0.01, "my_lr_scale": 1.0},
                    {"weight_decay": 0.0, "my_lr_scale": 2.0},
                ]
            )
        ],
    )

    callback.on_train_batch_start(trainer, object(), None, 0)

    expected_decay_lr = 1e-4 + (1e-5 - 1e-4) * (5 / 9)
    expected_lr = expected_decay_lr * (0.01 + 0.99 * 5 / 10)
    assert trainer.my_lr == pytest.approx(expected_lr)
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert trainer.optimizers[0].param_groups[1]["lr"] == pytest.approx(expected_lr * 2)
    assert trainer.optimizers[0].param_groups[0]["weight_decay"] == pytest.approx(0.01)
    trainer.my_log.close()


def test_train_callback_resume_global_step_keeps_sft_wsd_decay_position(tmp_path):
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="deepspeed_stage_3_offload",
            proj_dir=str(tmp_path),
            wandb="",
            my_timestamp="2026-06-09-12-30-00",
            run_name="sft-wsd-resume-test",
            epoch_begin=0,
            epoch_steps=10,
            epoch_count=2,
            warmup_steps=0,
            my_exit_tokens=0,
            ctx_len=16,
            real_bsz=1,
            lr_init=1e-4,
            lr_final=1e-5,
            lr_wsd_decay_iters=10,
            lr_wsd_decay_style="linear",
            weight_decay=0.0,
        )
    )

    trainer = SimpleNamespace(
        global_step=15,
        is_global_zero=True,
        strategy=SimpleNamespace(config={"zero_optimization": {"stage": 3}}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.0, "my_lr_scale": 1.0}])],
    )

    callback.on_train_batch_start(trainer, object(), None, 0)

    expected_lr = 1e-4 + (1e-5 - 1e-4) * (5 / 9)
    assert trainer.my_lr == pytest.approx(expected_lr)
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(expected_lr)
    trainer.my_log.close()

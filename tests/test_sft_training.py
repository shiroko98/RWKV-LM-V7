import sys
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train
from src import dataset as dataset_mod
from src import lr_schedule
from src import trainer as trainer_mod
from src.sft_split import (
    compute_sft_shuffled_split_indices,
    compute_sft_tail_eval_count,
    compute_sft_train_document_count,
)
from src.sft_binidx import EncodedDocument, write_documents
from src.sft_loss import masked_cross_entropy, masked_head_cross_entropy

_CALC_SFT_ONEPASS_SPEC = importlib.util.spec_from_file_location(
    "calc_sft_onepass_steps", ROOT / "scripts" / "calc_sft_onepass_steps.py"
)
calc_sft_onepass_steps = importlib.util.module_from_spec(_CALC_SFT_ONEPASS_SPEC)
assert _CALC_SFT_ONEPASS_SPEC.loader is not None
_CALC_SFT_ONEPASS_SPEC.loader.exec_module(calc_sft_onepass_steps)

_PROBE_SFT_STEPS_SPEC = importlib.util.spec_from_file_location(
    "probe_sft_binidx_steps", ROOT / "scripts" / "probe_sft_binidx_steps.py"
)
probe_sft_binidx_steps = importlib.util.module_from_spec(_PROBE_SFT_STEPS_SPEC)
assert _PROBE_SFT_STEPS_SPEC.loader is not None
sys.modules[_PROBE_SFT_STEPS_SPEC.name] = probe_sft_binidx_steps
_PROBE_SFT_STEPS_SPEC.loader.exec_module(probe_sft_binidx_steps)


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
        sft_eval_tail_ratio=0.0,
        sft_eval_tail_docs=0,
        sft_eval_every_n_steps=0,
        sft_eval_steps=0,
        sft_train_shuffle=0,
        sft_train_shuffle_seed=1234,
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


def test_sft_dataset_can_shuffle_train_documents_by_epoch_seed(tmp_path):
    prefix = str(tmp_path / "sft_shuffle")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in range(0, 70, 10)
        ],
    )
    args = make_sft_args(
        prefix,
        ctx_len=3,
        epoch_steps=4,
        sft_eval_tail_docs=2,
        sft_eval_steps=2,
        sft_train_shuffle=1,
        sft_train_shuffle_seed=7,
    )
    train_dataset = dataset_mod.MyDataset(args, sft_split="train")
    eval_dataset = dataset_mod.MyDataset(args, sft_split="eval")

    split_train, split_eval = compute_sft_shuffled_split_indices(7, 2, seed=7)
    epoch0 = np.random.default_rng(7).permutation(len(split_train)).tolist()
    epoch1 = np.random.default_rng(8).permutation(len(split_train)).tolist()

    train_x0, _, _ = train_dataset[0]
    train_x1, _, _ = train_dataset[1]
    assert int(train_x0[0].item() // 10) == int(split_train[epoch0[0]])
    assert int(train_x1[0].item() // 10) == int(split_train[epoch0[1]])

    train_dataset.real_epoch = 1
    epoch_x0, _, _ = train_dataset[0]
    assert int(epoch_x0[0].item() // 10) == int(split_train[epoch1[0]])
    assert train_dataset._sft_doc_index_from_sample(5, 1) == int(split_train[epoch1[1]])
    assert set(train_dataset.sft_doc_indices.tolist()).isdisjoint(set(eval_dataset.sft_doc_indices.tolist()))

    eval_x0, _, _ = eval_dataset[0]
    eval_x1, _, _ = eval_dataset[1]
    expected_eval_x0 = torch.tensor(
        [int(split_eval[0]) * 10, int(split_eval[0]) * 10 + 1, 65532],
        dtype=torch.long,
    )
    expected_eval_x1 = torch.tensor(
        [int(split_eval[1]) * 10, int(split_eval[1]) * 10 + 1, 65532],
        dtype=torch.long,
    )
    assert torch.equal(eval_x0, expected_eval_x0)
    assert torch.equal(eval_x1, expected_eval_x1)
    assert [int(split_eval[0]), int(split_eval[1])] != [5, 6]


def test_sft_tail_eval_split_excludes_eval_docs_from_train_and_reads_tail(tmp_path):
    prefix = str(tmp_path / "sft_tail_eval")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in [10, 20, 30, 40, 50]
        ],
    )
    args = make_sft_args(prefix, ctx_len=3, epoch_steps=4, sft_eval_tail_docs=2, sft_eval_steps=3)

    train_dataset = dataset_mod.MyDataset(args, sft_split="train")
    assert train_dataset.sft_doc_start == 0
    assert train_dataset.sft_doc_count == 3
    assert train_dataset.sft_eval_tail_count == 2

    x0, _, _ = train_dataset[0]
    x2, _, _ = train_dataset[2]
    x3, _, _ = train_dataset[3]
    assert torch.equal(x0, torch.tensor([10, 11, 65532], dtype=torch.long))
    assert torch.equal(x2, torch.tensor([30, 31, 65532], dtype=torch.long))
    assert torch.equal(x3, torch.tensor([10, 11, 65532], dtype=torch.long))

    eval_dataset = dataset_mod.MyDataset(args, sft_split="eval")
    assert len(eval_dataset) == 3
    assert eval_dataset.sft_doc_start == 3
    assert eval_dataset.sft_doc_count == 2

    eval_x0, _, _ = eval_dataset[0]
    eval_x1, _, _ = eval_dataset[1]
    eval_x2, _, _ = eval_dataset[2]
    assert torch.equal(eval_x0, torch.tensor([40, 41, 65532], dtype=torch.long))
    assert torch.equal(eval_x1, torch.tensor([50, 51, 65532], dtype=torch.long))
    assert torch.equal(eval_x2, torch.tensor([40, 41, 65532], dtype=torch.long))


def test_sft_tail_eval_can_overlap_with_full_train_docs(tmp_path):
    prefix = str(tmp_path / "sft_tail_eval_overlap")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in [10, 20, 30, 40, 50]
        ],
    )
    args = make_sft_args(prefix, ctx_len=3, epoch_steps=6, sft_eval_tail_docs=2, sft_eval_include_in_train=1)

    train_dataset = dataset_mod.MyDataset(args, sft_split="train")
    assert train_dataset.sft_doc_start == 0
    assert train_dataset.sft_doc_count == 5
    assert train_dataset.sft_eval_tail_count == 2

    x4, _, _ = train_dataset[4]
    x5, _, _ = train_dataset[5]
    assert torch.equal(x4, torch.tensor([50, 51, 65532], dtype=torch.long))
    assert torch.equal(x5, torch.tensor([10, 11, 65532], dtype=torch.long))


def test_sft_tail_eval_overlap_allows_full_eval_tail(tmp_path):
    prefix = str(tmp_path / "sft_tail_eval_full_overlap")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in [10, 20, 30]
        ],
    )
    args = make_sft_args(
        prefix,
        ctx_len=3,
        epoch_steps=3,
        sft_eval_tail_docs=3,
        sft_eval_include_in_train=1,
    )

    train_dataset = dataset_mod.MyDataset(args, sft_split="train")
    eval_dataset = dataset_mod.MyDataset(args, sft_split="eval")

    assert train_dataset.sft_doc_start == 0
    assert train_dataset.sft_doc_count == 3
    assert eval_dataset.sft_doc_start == 0
    assert eval_dataset.sft_doc_count == 3
    assert len(eval_dataset) == 3

    train_x0, _, _ = train_dataset[0]
    eval_x0, _, _ = eval_dataset[0]
    assert torch.equal(train_x0, torch.tensor([10, 11, 65532], dtype=torch.long))
    assert torch.equal(eval_x0, torch.tensor([10, 11, 65532], dtype=torch.long))


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

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([3]))
    with pytest.raises(ValueError, match="Unsupported SFT split"):
        dataset_mod.MyDataset(args(), sft_split="bad")

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([3]))
    with pytest.raises(ValueError, match="sft_train_shuffle"):
        dataset_mod.MyDataset(args(sft_train_shuffle=2))

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([3]))
    with pytest.raises(ValueError, match="sft_train_shuffle_seed"):
        dataset_mod.MyDataset(args(sft_train_shuffle_seed=-1))

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([3]))
    dataset = dataset_mod.MyDataset(args(sft_train_shuffle_seed=None))
    assert dataset.sft_train_shuffle_seed == 1234

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([3]))
    with pytest.raises(ValueError, match="eval split"):
        dataset_mod.MyDataset(args(), sft_split="eval")

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", lambda path: DummyMMap([3]))
    monkeypatch.setattr(dataset_mod, "compute_sft_tail_eval_count", lambda *_, **__: 1)
    with pytest.raises(ValueError, match="train split has no documents"):
        dataset_mod.MyDataset(args())


def test_pretrain_binidx_dataset_path_reads_magic_prime_schedule(monkeypatch):
    class DummyIndex:
        _dtype_size = 2

    class DummyPretrainMMap:
        _index = DummyIndex()

        def __init__(self, path):
            self._bin_buffer = bytes(10 * self._index._dtype_size)

        def get(self, idx, offset, length):
            return np.arange(offset, offset + length, dtype=np.int64)

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", DummyPretrainMMap)
    args = SimpleNamespace(
        vocab_size=65536,
        data_file="dummy",
        data_type="binidx",
        epoch_steps=40320,
        real_bsz=1,
        train_stage=0,
        ctx_len=2,
        magic_prime=5,
        epoch_begin=0,
        resume_epoch=0,
        resume_step_offset=0,
        micro_bsz=1,
        accumulate_grad_batches=4,
    )

    dataset = dataset_mod.MyDataset(args)
    assert len(dataset) == 40320
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([6, 7], dtype=torch.long))
    assert torch.equal(y, torch.tensor([7, 8], dtype=torch.long))


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


def test_checkpoint_name_and_resume_helpers_cover_deepspeed_branches(tmp_path):
    assert train.parse_epoch_checkpoint_name("rwkv-init.pth") == -1
    assert train.parse_epoch_checkpoint_name("rwkv-12.pth") == 12
    assert train.parse_epoch_checkpoint_name("rwkv-step-12.pth") is None
    assert train.parse_step_checkpoint_name("rwkv-step-34.pth") == 34
    assert train.parse_step_checkpoint_name("rwkv-34.pth") is None

    ds_dir = tmp_path / "rwkv-step-1.pth"
    ds_dir.mkdir()
    (ds_dir / "latest").write_text("global_step1", encoding="utf-8")
    assert train.is_deepspeed_checkpoint_dir(str(ds_dir)) is True
    assert train.resolve_resume_checkpoint_path(str(ds_dir), "deepspeed_stage_3_offload") == str(ds_dir)
    with pytest.raises(ValueError, match="DeepSpeed sharded"):
        train.resolve_resume_checkpoint_path(str(ds_dir), "auto")


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


def test_sft_one_pass_uses_train_docs_after_tail_eval_split(monkeypatch):
    args = SimpleNamespace(
        data_type="sft_binidx",
        data_file="dummy",
        sft_one_pass=1,
        epoch_steps=0,
        epoch_count=0,
        real_bsz=8,
        effective_bsz=32,
        accumulate_grad_batches=4,
        sft_eval_tail_ratio=0.0,
        sft_eval_tail_docs=1,
    )

    train.configure_epoch_schedule(args)
    monkeypatch.setattr(train, "count_binidx_documents", lambda prefix: 65)
    train.configure_sft_one_pass(args)
    train.configure_samples_per_epoch(args)
    train.configure_training_limits(args)

    assert args.sft_one_pass_documents == 64
    assert args.sft_eval_tail_documents == 1
    assert args.epoch_steps == 2
    assert args.epoch_count == 1
    assert args.max_epochs == 1
    assert args.samples_per_epoch == 64


def test_sft_one_pass_can_include_eval_tail_in_train_docs(monkeypatch):
    args = SimpleNamespace(
        data_type="sft_binidx",
        data_file="dummy",
        sft_one_pass=1,
        epoch_steps=0,
        epoch_count=0,
        real_bsz=8,
        effective_bsz=32,
        accumulate_grad_batches=4,
        sft_eval_tail_ratio=0.0,
        sft_eval_tail_docs=1,
        sft_eval_include_in_train=1,
    )

    train.configure_epoch_schedule(args)
    monkeypatch.setattr(train, "count_binidx_documents", lambda prefix: 65)
    train.configure_sft_one_pass(args)
    train.configure_samples_per_epoch(args)
    train.configure_training_limits(args)

    assert args.sft_one_pass_documents == 65
    assert args.sft_eval_tail_documents == 1
    assert args.epoch_steps == 3
    assert args.samples_per_epoch == 96


def test_sft_one_pass_can_eval_full_tail_when_train_uses_all_docs(monkeypatch):
    args = SimpleNamespace(
        data_type="sft_binidx",
        data_file="dummy",
        sft_one_pass=1,
        epoch_steps=0,
        epoch_count=0,
        real_bsz=8,
        effective_bsz=32,
        accumulate_grad_batches=4,
        sft_eval_tail_ratio=1.0,
        sft_eval_tail_docs=0,
        sft_eval_include_in_train=1,
    )

    train.configure_epoch_schedule(args)
    monkeypatch.setattr(train, "count_binidx_documents", lambda prefix: 65)
    train.configure_sft_one_pass(args)

    assert args.sft_one_pass_documents == 65
    assert args.sft_eval_tail_documents == 65
    assert args.epoch_steps == 3


def test_sft_one_pass_disabled_sets_default_metadata():
    args = SimpleNamespace(data_type="sft_binidx", sft_one_pass=0, epoch_steps=7, epoch_count=3)
    train.configure_sft_one_pass(args)
    assert args.sft_one_pass == 0
    assert args.sft_one_pass_documents == 0
    assert args.epoch_steps == 7
    assert args.epoch_count == 3


def test_sft_tail_eval_count_helper():
    assert compute_sft_tail_eval_count(100, tail_ratio=0.0, tail_docs=0) == 0
    assert compute_sft_tail_eval_count(100, tail_ratio=0.005, tail_docs=0) == 1
    assert compute_sft_tail_eval_count(100, tail_ratio=0.2, tail_docs=0) == 20
    assert compute_sft_tail_eval_count(100, tail_ratio=0.2, tail_docs=7) == 7
    assert compute_sft_train_document_count(100, tail_ratio=0.2, tail_docs=0) == 80

    with pytest.raises(ValueError, match="document_count"):
        compute_sft_tail_eval_count(0)
    with pytest.raises(ValueError, match="ratio"):
        compute_sft_tail_eval_count(100, tail_ratio=1.1)
    with pytest.raises(ValueError, match="tail_docs"):
        compute_sft_tail_eval_count(100, tail_docs=-1)
    with pytest.raises(ValueError, match="no training"):
        compute_sft_tail_eval_count(1, tail_ratio=0.5)
    with pytest.raises(ValueError, match="exceed"):
        compute_sft_tail_eval_count(10, tail_docs=11, require_train_docs=False)
    assert compute_sft_tail_eval_count(1, tail_ratio=1.0, require_train_docs=False) == 1

    train_indices, eval_indices = compute_sft_shuffled_split_indices(10, 2, seed=5)
    assert len(train_indices) == 8
    assert len(eval_indices) == 2
    assert set(train_indices.tolist()).isdisjoint(set(eval_indices.tolist()))
    assert eval_indices.tolist() != [8, 9]

    overlap_train, overlap_eval = compute_sft_shuffled_split_indices(
        10,
        10,
        eval_include_in_train=True,
        seed=5,
    )
    assert sorted(overlap_train.tolist()) == list(range(10))
    assert sorted(overlap_eval.tolist()) == list(range(10))

    full_train, empty_eval = compute_sft_shuffled_split_indices(10, 0, seed=5)
    assert sorted(full_train.tolist()) == list(range(10))
    assert empty_eval.tolist() == []

    with pytest.raises(ValueError, match="document_count"):
        compute_sft_shuffled_split_indices(0, 0, seed=5)
    with pytest.raises(ValueError, match="eval_count"):
        compute_sft_shuffled_split_indices(10, -1, seed=5)
    with pytest.raises(ValueError, match="eval_count"):
        compute_sft_shuffled_split_indices(10, 11, seed=5)
    with pytest.raises(ValueError, match="no training"):
        compute_sft_shuffled_split_indices(10, 10, seed=5)
    with pytest.raises(ValueError, match="seed"):
        compute_sft_shuffled_split_indices(10, 1, seed=-1)


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
    assert calc_sft_onepass_steps.count_documents(prefix) == 3
    assert calc_sft_onepass_steps.count_documents(prefix + ".idx") == 3
    assert calc_sft_onepass_steps.count_documents(prefix + ".bin") == 3


def test_calc_sft_onepass_schedule():
    schedule = calc_sft_onepass_steps.compute_onepass_schedule(
        65,
        num_nodes=1,
        devices=8,
        micro_bsz=1,
        accumulate_grad_batches=4,
        n_pass=2,
        ctx_len=5,
    )

    assert schedule["real_bsz"] == 8
    assert schedule["effective_bsz"] == 32
    assert schedule["epoch_steps"] == 3
    assert schedule["epoch_count"] == 2
    assert schedule["total_optimizer_steps"] == 6
    assert schedule["samples_per_epoch"] == 96
    assert schedule["extra_repeated_per_epoch"] == 31
    assert schedule["tokens_per_epoch"] == 480


def test_calc_sft_onepass_cli_reports_eval_split(tmp_path, capsys):
    prefix = str(tmp_path / "calc_cli")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in [10, 20, 30]
        ],
    )

    rc = calc_sft_onepass_steps.main(
        [
            prefix,
            "--num-nodes",
            "1",
            "--devices",
            "8",
            "--micro-bsz",
            "1",
            "--accumulate-grad-batches",
            "1",
            "--eval-tail-ratio",
            "1",
            "--eval-include-in-train",
            "1",
            "--n-pass",
            "1",
            "--ctx-len",
            "4",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "total_documents=3" in out
    assert "train_documents=3" in out
    assert "eval_documents=3" in out
    assert "eval_include_in_train=1" in out
    assert "epoch_steps=1" in out


def test_probe_sft_binidx_steps_matches_training_dataset_sampling(tmp_path):
    prefix = str(tmp_path / "probe_sampling")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in range(0, 160, 10)
        ],
    )
    args = make_sft_args(
        prefix,
        ctx_len=3,
        real_bsz=2,
        epoch_steps=2,
        accumulate_grad_batches=2,
    )
    dataset = dataset_mod.MyDataset(args)
    dataset.world_size = 2

    config = probe_sft_binidx_steps.BatchConfig(
        num_nodes=1,
        devices=2,
        micro_bsz=1,
        accumulate_grad_batches=2,
        epoch_steps=2,
    )
    split = probe_sft_binidx_steps.SplitInfo(
        total_documents=16,
        train_documents=16,
        eval_documents=0,
        eval_include_in_train=0,
    )

    probed = probe_sft_binidx_steps.doc_indices_for_optimizer_index(1, config, split)

    observed = []
    for idx in (2, 3):
        for rank in range(2):
            dataset.global_rank = rank
            x, _, _ = dataset[idx]
            observed.append(int(x[0].item() // 10))

    assert probed == observed


def test_probe_sft_binidx_steps_matches_training_dataset_shuffle_sampling(tmp_path):
    prefix = str(tmp_path / "probe_shuffle_sampling")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in range(0, 160, 10)
        ],
    )
    args = make_sft_args(
        prefix,
        ctx_len=3,
        real_bsz=2,
        epoch_steps=2,
        accumulate_grad_batches=2,
        sft_train_shuffle=1,
        sft_train_shuffle_seed=19,
    )
    dataset = dataset_mod.MyDataset(args)
    dataset.world_size = 2
    dataset.real_epoch = 1

    config = probe_sft_binidx_steps.BatchConfig(
        num_nodes=1,
        devices=2,
        micro_bsz=1,
        accumulate_grad_batches=2,
        epoch_steps=2,
        epoch_begin=1,
        sft_train_shuffle=1,
        sft_train_shuffle_seed=19,
    )
    split = probe_sft_binidx_steps.SplitInfo(
        total_documents=16,
        train_documents=16,
        eval_documents=0,
        eval_include_in_train=0,
    )

    probed = probe_sft_binidx_steps.doc_indices_for_optimizer_index(1, config, split)

    observed = []
    for idx in (2, 3):
        for rank in range(2):
            dataset.global_rank = rank
            x, _, _ = dataset[idx]
            observed.append(int(x[0].item() // 10))

    assert probed == observed


def test_shuffled_sft_split_resume_and_multirank_sampling_matches_probe(tmp_path):
    prefix = str(tmp_path / "probe_shuffle_split_resume")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in range(0, 120, 10)
        ],
    )
    args = make_sft_args(
        prefix,
        ctx_len=3,
        real_bsz=2,
        epoch_steps=3,
        accumulate_grad_batches=2,
        sft_eval_tail_docs=3,
        sft_eval_steps=3,
        sft_train_shuffle=1,
        sft_train_shuffle_seed=23,
    )
    train_dataset = dataset_mod.MyDataset(args, sft_split="train")
    eval_dataset = dataset_mod.MyDataset(args, sft_split="eval")
    train_dataset.world_size = 2
    train_dataset.real_epoch = 2
    train_dataset.step_offset = 1

    split_train, split_eval = compute_sft_shuffled_split_indices(12, 3, seed=23)
    assert train_dataset.sft_doc_indices.tolist() == split_train.tolist()
    assert eval_dataset.sft_doc_indices.tolist() == split_eval.tolist()
    assert set(split_train.tolist()).isdisjoint(set(split_eval.tolist()))
    assert split_eval.tolist() != [9, 10, 11]

    config = probe_sft_binidx_steps.BatchConfig(
        num_nodes=1,
        devices=2,
        micro_bsz=1,
        accumulate_grad_batches=2,
        epoch_steps=3,
        epoch_begin=2,
        sft_train_shuffle=1,
        sft_train_shuffle_seed=23,
    )
    split = probe_sft_binidx_steps.compute_split_info(
        12,
        eval_tail_ratio=0.0,
        eval_tail_docs=3,
        eval_include_in_train=0,
        sft_train_shuffle=1,
        sft_train_shuffle_seed=23,
    )

    probed = probe_sft_binidx_steps.doc_indices_for_optimizer_index(1, config, split)
    observed = []
    for idx in (0, 1):
        for rank in range(2):
            train_dataset.global_rank = rank
            x, _, _ = train_dataset[idx]
            observed.append(int(x[0].item() // 10))

    assert probed == observed


def test_probe_sft_binidx_steps_cli_reports_windows_and_overlap(tmp_path, capsys):
    prefix = str(tmp_path / "probe_cli")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1, base + 2], loss_mask=[0, 1, 1])
            for base in range(0, 120, 10)
        ],
    )

    rc = probe_sft_binidx_steps.main(
        [
            prefix,
            "--ctx-len",
            "4",
            "--num-nodes",
            "1",
            "--devices",
            "2",
            "--micro-bsz",
            "1",
            "--accumulate-grad-batches",
            "2",
            "--epoch-steps",
            "3",
            "--probe-steps",
            "1,4",
            "--window-steps",
            "1",
            "--show-docs",
            "1",
        ]
    )

    assert rc == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["kind"] == "config"
    windows = [record for record in records if record["kind"] == "window"]
    assert [record["logged_step"] for record in windows] == [1, 4]
    assert windows[0]["docs"] == 4
    assert windows[0]["unique_docs"] == 4
    comparisons = [record for record in records if record["kind"] == "comparison"]
    assert comparisons[0]["left_step"] == 1
    assert comparisons[0]["right_step"] == 4
    assert comparisons[0]["common_docs"] == 4
    assert comparisons[0]["ordered_equal"] is True


def test_probe_sft_binidx_step_helpers_cover_ranges_and_errors(tmp_path):
    prefix = str(tmp_path / "probe_count")
    write_documents(prefix, [EncodedDocument(input_ids=[1, 2], loss_mask=[0, 1])])

    assert probe_sft_binidx_steps.normalize_prefix(prefix + ".idx") == prefix
    assert probe_sft_binidx_steps.normalize_prefix(prefix + ".bin") == prefix
    assert probe_sft_binidx_steps.count_documents(prefix + ".idx") == 1
    with pytest.raises(FileNotFoundError):
        probe_sft_binidx_steps.count_documents(str(tmp_path / "missing"))

    split = probe_sft_binidx_steps.compute_split_info(
        10,
        eval_tail_ratio=0.2,
        eval_tail_docs=0,
        eval_include_in_train=1,
    )
    assert split.train_documents == 10
    assert split.eval_documents == 2
    assert probe_sft_binidx_steps.auto_epoch_steps(9, 4) == 3
    with pytest.raises(ValueError, match="train_documents"):
        probe_sft_binidx_steps.auto_epoch_steps(0, 4)
    with pytest.raises(ValueError, match="effective_bsz"):
        probe_sft_binidx_steps.auto_epoch_steps(1, 0)

    assert probe_sft_binidx_steps.parse_probe_steps("1, 3:7:2, 10-12, , -2") == [1, 3, 5, 7, 10, 11, 12, -2]
    with pytest.raises(ValueError, match="Invalid probe"):
        probe_sft_binidx_steps.parse_probe_steps("1:2:3:4")
    with pytest.raises(ValueError, match="stride"):
        probe_sft_binidx_steps.parse_probe_steps("1:2:0")
    with pytest.raises(ValueError, match="At least one"):
        probe_sft_binidx_steps.parse_probe_steps(",")

    assert probe_sft_binidx_steps.logged_step_to_optimizer_index(0, "zero") == 0
    with pytest.raises(ValueError, match="step_base"):
        probe_sft_binidx_steps.logged_step_to_optimizer_index(1, "bad")
    with pytest.raises(ValueError, match="negative"):
        probe_sft_binidx_steps.logged_step_to_optimizer_index(0, "one")

    config = probe_sft_binidx_steps.BatchConfig(1, 2, 1, 2, 3, epoch_begin=1)
    assert probe_sft_binidx_steps.sample_indices_for_optimizer_index(3, config) == [24, 25, 26, 27]
    with pytest.raises(ValueError, match="optimizer_index"):
        probe_sft_binidx_steps.sample_indices_for_optimizer_index(-1, config)
    with pytest.raises(ValueError, match="epoch_steps"):
        probe_sft_binidx_steps.sample_indices_for_optimizer_index(
            0,
            probe_sft_binidx_steps.BatchConfig(1, 1, 1, 1, 0),
        )
    with pytest.raises(ValueError, match="no documents"):
        probe_sft_binidx_steps.doc_indices_for_optimizer_index(
            0,
            config,
            probe_sft_binidx_steps.SplitInfo(0, 0, 0, 0),
        )
    with pytest.raises(ValueError, match="window_steps"):
        probe_sft_binidx_steps.window_doc_indices(
            1,
            window_steps=0,
            step_base="one",
            config=config,
            split=probe_sft_binidx_steps.SplitInfo(4, 4, 0, 0),
        )
    with pytest.raises(ValueError, match="positive"):
        probe_sft_binidx_steps.validate_positive("x", 0)


def test_probe_sft_binidx_doc_stats_text_decode_and_validation(tmp_path):
    prefix = str(tmp_path / "probe_stats")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[1, 2, 65532, 65532], loss_mask=[0, 1, 0, 0]),
            EncodedDocument(input_ids=[3, 4, 5, 6, 7], loss_mask=[0, 1, 1, 1, 1]),
        ],
    )
    vocab = ROOT / "data" / "tokenizer" / "rwkv_vocab_v20230424.txt"
    probe = probe_sft_binidx_steps.SftBinidxProbe(
        prefix,
        ctx_len=3,
        vocab_path=str(vocab),
        show_text_chars=8,
    )

    first = probe.doc_stats(0)
    assert first["train_tokens"] == 1
    assert first["tail_padding"] == 2
    assert first["mask_density"] == pytest.approx(1 / 3)
    assert "trainable_excerpt" in first

    second = probe.doc_stats(1)
    assert second["too_long"] is True
    assert second["target_tokens"] == 3

    bad_mask_prefix = str(tmp_path / "probe_bad_mask")
    write_documents(
        bad_mask_prefix,
        [
            EncodedDocument(input_ids=[1, 2], loss_mask=[0, 1]),
            EncodedDocument(input_ids=[3, 4], loss_mask=[0, 1]),
            EncodedDocument(input_ids=[5, 6], loss_mask=[0, 1]),
        ],
    )
    with pytest.raises(ValueError, match="counts differ"):
        probe_sft_binidx_steps.SftBinidxProbe(prefix, ctx_len=3, mask_prefix=bad_mask_prefix)

    empty = probe_sft_binidx_steps.summarize_window(1, [], [], step_base="one", window_steps=1)
    assert empty["doc_min"] is None
    assert empty["train_tokens_mean"] == 0.0
    empty_cmp = probe_sft_binidx_steps.compare_windows(
        {"logged_step": 1, "window_hash": "a"},
        [],
        {"logged_step": 2, "window_hash": "a"},
        [],
    )
    assert empty_cmp["jaccard"] == 0.0
    assert empty_cmp["window_hash_equal"] is True

    class BadTokenizer:
        def decode(self, tokens):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")

        def decodeBytes(self, tokens):
            return b"\xff"

    assert probe_sft_binidx_steps.decode_trainable_excerpt(
        BadTokenizer(),
        np.array([1, 2], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        4,
    ) == "�"
    assert probe_sft_binidx_steps.decode_trainable_excerpt(
        BadTokenizer(),
        np.array([1, 2], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
        4,
    ) == ""


def test_probe_sft_binidx_cli_writes_jsonl_and_auto_epoch_steps(tmp_path, capsys):
    prefix = str(tmp_path / "probe_jsonl")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[base, base + 1], loss_mask=[0, 1])
            for base in range(0, 80, 10)
        ],
    )
    jsonl_out = tmp_path / "probe.jsonl"

    rc = probe_sft_binidx_steps.main(
        [
            prefix + ".bin",
            "--ctx-len",
            "3",
            "--num-nodes",
            "1",
            "--devices",
            "2",
            "--micro-bsz",
            "1",
            "--accumulate-grad-batches",
            "2",
            "--eval-tail-ratio",
            "0.25",
            "--probe-steps",
            "0",
            "--step-base",
            "zero",
            "--show-docs",
            "0",
            "--jsonl-out",
            str(jsonl_out),
        ]
    )

    assert rc == 0
    stdout_records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    file_records = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").splitlines()]
    assert stdout_records == file_records
    assert stdout_records[0]["epoch_steps"] == 2
    assert stdout_records[0]["train_documents"] == 6


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


def test_validate_sft_train_shuffle_settings_normalizes_and_rejects_invalid_values():
    args = SimpleNamespace(sft_train_shuffle="1", sft_train_shuffle_seed="42")
    train.validate_sft_train_shuffle_settings(args)
    assert args.sft_train_shuffle == 1
    assert args.sft_train_shuffle_seed == 42

    args = SimpleNamespace(sft_train_shuffle=0, sft_train_shuffle_seed=None)
    train.validate_sft_train_shuffle_settings(args)
    assert args.sft_train_shuffle == 0
    assert args.sft_train_shuffle_seed == 1234

    with pytest.raises(ValueError, match="sft_train_shuffle"):
        train.validate_sft_train_shuffle_settings(SimpleNamespace(sft_train_shuffle=2, sft_train_shuffle_seed=0))
    with pytest.raises(ValueError, match="sft_train_shuffle_seed"):
        train.validate_sft_train_shuffle_settings(SimpleNamespace(sft_train_shuffle=1, sft_train_shuffle_seed=-1))
    with pytest.raises(ValueError, match="sft_train_shuffle_seed"):
        train.validate_sft_train_shuffle_settings(SimpleNamespace(sft_train_shuffle=1, sft_train_shuffle_seed="bad"))


def test_validate_sft_eval_settings():
    train.validate_sft_eval_settings(
        SimpleNamespace(
            data_type="sft_binidx",
            sft_eval_every_n_steps=1200,
            sft_eval_steps=16,
            sft_eval_tail_ratio=0.005,
            sft_eval_tail_docs=0,
            sft_eval_include_in_train=0,
        )
    )
    train.validate_sft_eval_settings(
        SimpleNamespace(
            data_type="sft_binidx",
            sft_eval_every_n_steps=0,
            sft_eval_steps=0,
            sft_eval_tail_ratio=0.0,
            sft_eval_tail_docs=0,
            sft_eval_include_in_train=1,
        )
    )

    with pytest.raises(ValueError, match="sft_binidx"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="binidx",
                sft_eval_every_n_steps=1,
                sft_eval_steps=1,
                sft_eval_tail_ratio=0.1,
                sft_eval_tail_docs=0,
                sft_eval_include_in_train=0,
            )
        )
    with pytest.raises(ValueError, match="sft_eval_steps"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="sft_binidx",
                sft_eval_every_n_steps=1,
                sft_eval_steps=0,
                sft_eval_tail_ratio=0.1,
                sft_eval_tail_docs=0,
                sft_eval_include_in_train=0,
            )
        )
    with pytest.raises(ValueError, match="tail"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="sft_binidx",
                sft_eval_every_n_steps=1,
                sft_eval_steps=1,
                sft_eval_tail_ratio=0.0,
                sft_eval_tail_docs=0,
                sft_eval_include_in_train=0,
            )
        )
    with pytest.raises(ValueError, match="include"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="sft_binidx",
                sft_eval_every_n_steps=0,
                sft_eval_steps=0,
                sft_eval_tail_ratio=0.0,
                sft_eval_tail_docs=0,
                sft_eval_include_in_train=2,
            )
        )
    with pytest.raises(ValueError, match="sft_eval_every_n_steps"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="sft_binidx",
                sft_eval_every_n_steps=-1,
                sft_eval_steps=0,
                sft_eval_tail_ratio=0.0,
                sft_eval_tail_docs=0,
                sft_eval_include_in_train=0,
            )
        )
    with pytest.raises(ValueError, match="sft_eval_steps"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="sft_binidx",
                sft_eval_every_n_steps=0,
                sft_eval_steps=-1,
                sft_eval_tail_ratio=0.0,
                sft_eval_tail_docs=0,
                sft_eval_include_in_train=0,
            )
        )
    with pytest.raises(ValueError, match="sft_eval_tail_docs"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="sft_binidx",
                sft_eval_every_n_steps=0,
                sft_eval_steps=0,
                sft_eval_tail_ratio=0.0,
                sft_eval_tail_docs=-1,
                sft_eval_include_in_train=0,
            )
        )
    with pytest.raises(ValueError, match="sft_eval_tail_ratio"):
        train.validate_sft_eval_settings(
            SimpleNamespace(
                data_type="sft_binidx",
                sft_eval_every_n_steps=0,
                sft_eval_steps=0,
                sft_eval_tail_ratio=1.0,
                sft_eval_tail_docs=0,
                sft_eval_include_in_train=0,
            )
        )


def test_configure_deepspeed_zero3_config_applies_sft_tuning_options():
    args = SimpleNamespace(
        strategy="deepspeed_stage_3_offload",
        ds_bucket_mb=128,
        ds_offload_pin_memory=1,
        ds_stage3_param_persistence_threshold=100000,
        ds_stage3_prefetch_bucket_size=20000000,
        ds_stage3_max_live_parameters=1000000000,
    )
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "offload_param": {"device": "cpu", "pin_memory": False},
        }
    }

    train.configure_deepspeed_zero3_config(args, config)
    zero = config["zero_optimization"]

    assert zero["allgather_bucket_size"] == 128_000_000
    assert zero["reduce_bucket_size"] == 128_000_000
    assert zero["offload_optimizer"]["pin_memory"] is True
    assert zero["offload_param"]["pin_memory"] is True
    assert zero["stage3_param_persistence_threshold"] == 100000
    assert zero["stage3_prefetch_bucket_size"] == 20000000
    assert zero["stage3_max_live_parameters"] == 1000000000


def test_configure_deepspeed_zero3_config_keeps_disabled_options_unchanged():
    non_ds_config = {"zero_optimization": {"allgather_bucket_size": 1}}
    train.configure_deepspeed_zero3_config(SimpleNamespace(strategy="ddp", ds_bucket_mb=256), non_ds_config)
    assert non_ds_config["zero_optimization"]["allgather_bucket_size"] == 1

    args = SimpleNamespace(
        strategy="deepspeed_stage_3_offload",
        ds_bucket_mb=None,
        ds_offload_pin_memory=None,
        ds_stage3_param_persistence_threshold=None,
        ds_stage3_prefetch_bucket_size=None,
        ds_stage3_max_live_parameters=None,
    )
    config = {
        "zero_optimization": {
            "offload_optimizer": {"pin_memory": False},
            "offload_param": {"pin_memory": True},
        }
    }
    train.configure_deepspeed_zero3_config(args, config)
    zero = config["zero_optimization"]
    assert "allgather_bucket_size" not in zero
    assert "stage3_prefetch_bucket_size" not in zero
    assert zero["offload_optimizer"]["pin_memory"] is False
    assert zero["offload_param"]["pin_memory"] is True


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"ds_bucket_mb": -1}, "ds_bucket_mb"),
        ({"ds_bucket_mb": "bad"}, "ds_bucket_mb"),
        ({"ds_offload_pin_memory": 2}, "ds_offload_pin_memory"),
        ({"ds_stage3_param_persistence_threshold": -2}, "ds_stage3_param_persistence_threshold"),
        ({"ds_stage3_prefetch_bucket_size": -2}, "ds_stage3_prefetch_bucket_size"),
        ({"ds_stage3_max_live_parameters": -2}, "ds_stage3_max_live_parameters"),
    ],
)
def test_configure_deepspeed_zero3_config_validates_values(overrides, match):
    base = dict(
        strategy="deepspeed_stage_3_offload",
        ds_bucket_mb=64,
        ds_offload_pin_memory=-1,
        ds_stage3_param_persistence_threshold=-1,
        ds_stage3_prefetch_bucket_size=-1,
        ds_stage3_max_live_parameters=-1,
    )
    base.update(overrides)
    with pytest.raises(ValueError, match=match):
        train.configure_deepspeed_zero3_config(SimpleNamespace(**base), {"zero_optimization": {}})


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


def test_build_wandb_train_metrics_uses_clear_names():
    args = SimpleNamespace(ctx_len=16, real_bsz=4, effective_bsz=8)
    trainer = SimpleNamespace(my_loss=1.25, my_epoch_loss=1.5, my_lr=2e-5, my_wd=0.001)

    metrics = trainer_mod.build_wandb_train_metrics(
        args,
        trainer,
        real_step=3,
        token_per_optimizer_step=128,
        t_cost=2.0,
        kt_s=0.256,
        grad_norm=7.5,
    )

    assert metrics["train/loss"] == pytest.approx(1.25)
    assert metrics["train/epoch_loss"] == pytest.approx(1.5)
    assert metrics["train/lr"] == pytest.approx(2e-5)
    assert metrics["train/weight_decay"] == pytest.approx(0.001)
    assert metrics["train/samples"] == 24
    assert metrics["train/tokens"] == 384
    assert metrics["train/tokens_b"] == pytest.approx(384 / 1e9)
    assert metrics["train/step"] == 3
    assert metrics["perf/iteration_time_sec"] == pytest.approx(2.0)
    assert metrics["perf/optimizer_steps_per_sec"] == pytest.approx(0.5)
    assert metrics["perf/tokens_per_sec"] == pytest.approx(64.0)
    assert metrics["perf/ktokens_per_sec"] == pytest.approx(0.256)
    assert metrics["perf/samples_per_sec"] == pytest.approx(4.0)
    assert metrics["train/grad_norm"] == pytest.approx(7.5)
    assert "Gtokens" not in metrics


def test_get_global_grad_norm_prefers_strategy_value_and_skips_deepspeed_fallback():
    trainer = SimpleNamespace(
        strategy=SimpleNamespace(model=SimpleNamespace(get_global_grad_norm=lambda: torch.tensor(3.5)))
    )
    assert trainer_mod.get_global_grad_norm(trainer, object()) == pytest.approx(3.5)

    trainer = SimpleNamespace(strategy="deepspeed_stage_3_offload")
    module = torch.nn.Linear(2, 2)
    loss = module(torch.ones(1, 2)).sum()
    loss.backward()
    assert trainer_mod.get_global_grad_norm(trainer, module) is None


def test_get_global_grad_norm_computes_local_non_deepspeed_norm():
    trainer = SimpleNamespace(strategy="")
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    loss = module(torch.ones(1, 2)).sum()
    loss.backward()

    assert trainer_mod.get_global_grad_norm(trainer, module) == pytest.approx(2 ** 0.5)


def test_train_callback_before_optimizer_step_accepts_optimizer_idx(tmp_path, monkeypatch):
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="",
            proj_dir=str(tmp_path),
            wandb="",
            my_timestamp="2026-06-11-15-30-00",
            run_name="grad-norm-hook-test",
        )
    )
    monkeypatch.setattr(trainer_mod, "get_global_grad_norm", lambda trainer, pl_module: 4.25)
    trainer = SimpleNamespace()

    callback.on_before_optimizer_step(trainer, object(), object(), 0)

    assert trainer.my_grad_norm == pytest.approx(4.25)


def test_train_callback_runs_save_and_eval_on_same_step(tmp_path, monkeypatch):
    events = []

    def fake_save(args, trainer, pl_module, file_name):
        events.append(("save", Path(file_name).name))

    def fake_eval(self, trainer, pl_module, real_step):
        events.append(("eval", real_step))

    monkeypatch.setattr(trainer_mod, "save_train_checkpoint", fake_save)
    monkeypatch.setattr(trainer_mod.train_callback, "_run_sft_eval", fake_eval)

    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="",
            proj_dir=str(tmp_path),
            magic_prime=0,
            save_every_n_steps=10,
            save_at_step=0,
            sft_eval_every_n_steps=10,
            ctx_len=16,
            real_bsz=1,
            effective_bsz=1,
            epoch_begin=0,
            epoch_steps=100,
            wandb="",
        ),
        eval_loader=object(),
    )

    trainer = SimpleNamespace(
        global_step=10,
        is_global_zero=True,
        my_time_ns=0,
        my_loss_all=torch.tensor([1.0]),
        my_loss_sum=0.0,
        my_loss_count=0,
        my_lr=1e-4,
        my_wd=0.0,
    )
    callback.log = lambda *args, **kwargs: None

    callback.on_train_batch_end(trainer, object(), None, None, 0)

    assert events == [("save", "rwkv-step-10.pth"), ("eval", 10)]


def test_train_callback_wandb_logs_accumulated_loss_once_at_completed_step(tmp_path):
    class FakeWandb:
        def __init__(self):
            self.records = []

        def log(self, values, step):
            self.records.append((values, step))

    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="",
            proj_dir=str(tmp_path),
            magic_prime=0,
            save_every_n_steps=0,
            save_at_step=0,
            sft_eval_every_n_steps=0,
            ctx_len=16,
            real_bsz=8,
            effective_bsz=32,
            epoch_begin=0,
            epoch_steps=1000,
            warmup_steps=0,
            my_exit_tokens=0,
            lr_init=1e-4,
            lr_final=1e-5,
            lr_wsd_decay_iters=0,
            lr_wsd_decay_style="cosine",
            weight_decay=0.0,
            wandb="enabled",
            run_name="accum-loss-test",
            my_timestamp="2026-06-23-11-30-00",
        )
    )
    callback.log = lambda *args, **kwargs: None

    trainer = SimpleNamespace(
        global_step=350,
        is_global_zero=True,
        strategy=SimpleNamespace(config={}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.0, "my_lr_scale": 1.0}])],
        my_wandb=FakeWandb(),
    )

    for loss in (1.0, 2.0, 3.0):
        callback.on_train_batch_start(trainer, object(), None, 0)
        trainer.my_loss_all = torch.tensor([loss])
        callback.on_train_batch_end(trainer, object(), None, None, 0)

    assert trainer.my_wandb.records == []

    callback.on_train_batch_start(trainer, object(), None, 0)
    trainer.global_step = 351
    trainer.my_loss_all = torch.tensor([4.0])
    callback.on_train_batch_end(trainer, object(), None, None, 0)

    assert len(trainer.my_wandb.records) == 1
    values, step = trainer.my_wandb.records[0]
    assert step == 351
    assert values["train/step"] == 351
    assert values["train/loss"] == pytest.approx(2.5)
    assert values["train/samples"] == 351 * 32
    trainer.my_log.close()


def test_train_callback_progress_metrics_log_once_at_completed_optimizer_step(tmp_path, monkeypatch):
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="",
            proj_dir=str(tmp_path),
            magic_prime=0,
            save_every_n_steps=0,
            save_at_step=0,
            sft_eval_every_n_steps=0,
            ctx_len=16,
            real_bsz=8,
            effective_bsz=32,
            epoch_begin=0,
            epoch_steps=1000,
            warmup_steps=0,
            my_exit_tokens=0,
            lr_init=1e-4,
            lr_final=1e-5,
            lr_wsd_decay_iters=0,
            lr_wsd_decay_style="cosine",
            weight_decay=0.0,
            wandb="",
            run_name="progress-step-test",
            my_timestamp="2026-06-24-09-00-00",
        )
    )
    logged = []
    callback.log = lambda name, value, **kwargs: logged.append((name, value, kwargs))
    times = iter([1_000_000_000, 3_000_000_000])
    monkeypatch.setattr(trainer_mod.time, "time_ns", lambda: next(times))

    trainer = SimpleNamespace(
        global_step=20,
        is_global_zero=True,
        strategy=SimpleNamespace(config={}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.0, "my_lr_scale": 1.0}])],
        progress_bar_metrics={},
    )

    callback.on_train_batch_start(trainer, object(), None, 0)
    trainer.my_loss_all = torch.tensor([1.0])
    callback.on_train_batch_end(trainer, object(), None, None, 0)
    assert logged == []
    assert trainer.progress_bar_metrics == {}

    callback.on_train_batch_start(trainer, object(), None, 0)
    trainer.global_step = 21
    trainer.my_loss_all = torch.tensor([3.0])
    callback.on_train_batch_end(trainer, object(), None, None, 0)

    assert logged == []
    assert trainer.progress_bar_metrics["REAL it/s"] == pytest.approx(0.5)
    assert trainer.progress_bar_metrics["Kt/s"] == pytest.approx((16 * 32) / 2.0 / 1000)
    assert trainer.progress_bar_metrics["lr"] == pytest.approx(1e-4)
    assert trainer.progress_bar_metrics["loss"] == pytest.approx(2.0)
    trainer.my_log.close()


def test_train_callback_does_not_repeat_save_or_eval_at_resume_step(tmp_path, monkeypatch):
    events = []

    def fake_save(args, trainer, pl_module, file_name):
        events.append(("save", Path(file_name).name))

    def fake_eval(self, trainer, pl_module, real_step):
        events.append(("eval", real_step))

    monkeypatch.setattr(trainer_mod, "save_train_checkpoint", fake_save)
    monkeypatch.setattr(trainer_mod.train_callback, "_run_sft_eval", fake_eval)

    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="sft_binidx",
            strategy="",
            proj_dir=str(tmp_path),
            magic_prime=0,
            save_every_n_steps=350,
            save_at_step=0,
            sft_eval_every_n_steps=350,
            ctx_len=16,
            real_bsz=8,
            effective_bsz=32,
            epoch_begin=0,
            epoch_steps=1000,
            warmup_steps=0,
            my_exit_tokens=0,
            lr_init=1e-4,
            lr_final=1e-5,
            lr_wsd_decay_iters=0,
            lr_wsd_decay_style="cosine",
            weight_decay=0.0,
            wandb="",
            run_name="resume-save-test",
            my_timestamp="2026-06-23-11-35-00",
        ),
        eval_loader=object(),
    )
    callback.log = lambda *args, **kwargs: None

    trainer = SimpleNamespace(
        global_step=350,
        is_global_zero=True,
        strategy=SimpleNamespace(config={}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.0, "my_lr_scale": 1.0}])],
        my_loss_all=torch.tensor([1.0]),
    )

    callback.on_train_batch_start(trainer, object(), None, 0)
    callback.on_train_batch_end(trainer, object(), None, None, 0)

    assert events == []
    trainer.my_log.close()


def test_train_callback_sft_eval_logs_mask_token_weighted_loss(tmp_path, monkeypatch):
    batches = [
        (
            torch.zeros((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
            torch.tensor([[1.0, 0.0, 0.0]]),
        ),
        (
            torch.zeros((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
            torch.tensor([[1.0, 1.0, 1.0]]),
        ),
    ]
    losses = [torch.tensor(2.0), torch.tensor(4.0)]

    class FakeModule:
        device = torch.device("cpu")
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def training_step(self, batch, batch_idx):
            return losses[batch_idx]

    class FakeWandb:
        def __init__(self):
            self.records = []

        def log(self, values, step):
            self.records.append((values, step))

    class FakeEvalLoader:
        dataset = SimpleNamespace()

        def __iter__(self):
            return iter(batches)

    monkeypatch.setattr(trainer_mod, "strategy_barrier", lambda trainer: None)

    args = SimpleNamespace(
        data_type="sft_binidx",
        proj_dir=str(tmp_path),
        wandb="enabled",
        run_name="eval-test",
        my_timestamp="2026-06-11-12-00-00",
        sft_eval_steps=2,
    )
    callback = trainer_mod.train_callback(args, eval_loader=SimpleNamespace(dataset=SimpleNamespace()))
    trainer = SimpleNamespace(
        global_rank=0,
        world_size=1,
        is_global_zero=True,
        strategy=SimpleNamespace(barrier=lambda: None),
        my_log=open(tmp_path / "train_log.txt", "a"),
        my_wandb=FakeWandb(),
    )
    callback.eval_loader = FakeEvalLoader()

    callback._run_sft_eval(trainer, FakeModule(), real_step=10)
    trainer.my_log.close()

    values, step = trainer.my_wandb.records[0]
    assert step == 10
    assert values["eval/loss"] == pytest.approx((2.0 * 1 + 4.0 * 3) / 4)
    assert values["eval/mask_tokens"] == 4
    assert values["eval/docs"] == 2


def test_train_callback_sft_eval_prefers_model_eval_step(tmp_path, monkeypatch):
    batch = (
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
        torch.tensor([[1.0, 1.0]]),
    )
    calls = []

    class FakeEvalLoader:
        dataset = SimpleNamespace()

        def __iter__(self):
            return iter([batch])

    class FakeModule:
        device = torch.device("cpu")
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def sft_eval_step(self, batch, batch_idx):
            calls.append(("eval", batch_idx))
            return torch.tensor(3.0)

        def training_step(self, batch, batch_idx):
            calls.append(("train", batch_idx))
            return torch.tensor(9.0)

    monkeypatch.setattr(trainer_mod, "strategy_barrier", lambda trainer: None)

    args = SimpleNamespace(
        data_type="sft_binidx",
        proj_dir=str(tmp_path),
        wandb="",
        run_name="eval-step-test",
        my_timestamp="2026-06-11-12-10-00",
        sft_eval_steps=1,
    )
    callback = trainer_mod.train_callback(args, eval_loader=FakeEvalLoader())
    trainer = SimpleNamespace(
        global_rank=0,
        world_size=1,
        is_global_zero=True,
        strategy=SimpleNamespace(barrier=lambda: None),
        my_log=open(tmp_path / "train_log.txt", "a"),
    )

    callback._run_sft_eval(trainer, FakeModule(), real_step=12)
    trainer.my_log.close()

    assert calls == [("eval", 0)]


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

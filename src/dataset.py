########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .sft_split import compute_sft_shuffled_split_indices, compute_sft_tail_eval_count

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class MyDataset(Dataset):
    def __init__(self, args, *, sft_split="train"):
        self.args = args
        self.sft_split = sft_split

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_type = getattr(args, "data_type", "binidx")
        self.mask_data = None
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.accumulate_grad_batches = int(getattr(args, "accumulate_grad_batches", 1) or 1)
        default_samples_per_epoch = args.epoch_steps * args.real_bsz
        if self.data_type == "sft_binidx":
            default_samples_per_epoch *= self.accumulate_grad_batches
        self.samples_per_epoch = int(getattr(args, "samples_per_epoch", default_samples_per_epoch))
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.real_epoch = getattr(args, "resume_epoch", args.epoch_begin)
        self.step_offset = getattr(args, "resume_step_offset", 0)
        self._sft_shuffle_epoch = None
        self._sft_shuffle_permutation = None
        self.sft_doc_indices = None

        if self.data_type == "binidx":
            assert self.samples_per_epoch == 40320
            dataset_slot = self.data_size // args.ctx_len
            assert is_prime(args.magic_prime)
            assert args.magic_prime % 3 == 2
            assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1
        elif self.data_type == "sft_binidx":
            if len(self.data) == 0:
                raise ValueError("SFT token dataset must contain at least one document.")
            if self.sft_split not in {"train", "eval"}:
                raise ValueError(f"Unsupported SFT split: {self.sft_split}")
            mask_file = getattr(args, "sft_mask_file", "") or f"{args.data_file}.mask"
            self.mask_data = MMapIndexedDataset(mask_file)
            if len(self.mask_data) != len(self.data):
                raise ValueError(
                    f"SFT token and mask datasets must have the same document count: "
                    f"{len(self.data)} != {len(self.mask_data)}."
                )
            if not np.array_equal(self.data.sizes, self.mask_data.sizes):
                raise ValueError("SFT token and mask datasets must have identical document sizes.")
            if args.ctx_len <= 0:
                raise ValueError("ctx_len must be positive for sft_binidx training.")
            self.sft_pad_token_id = int(getattr(args, "sft_pad_token_id", 65532))
            self.sft_train_shuffle = int(getattr(args, "sft_train_shuffle", 0) or 0)
            if self.sft_train_shuffle not in (0, 1):
                raise ValueError("sft_train_shuffle must be 0 or 1.")
            shuffle_seed = getattr(args, "sft_train_shuffle_seed", 1234)
            if shuffle_seed is None:
                shuffle_seed = 1234
            self.sft_train_shuffle_seed = int(shuffle_seed)
            if self.sft_train_shuffle_seed < 0:
                raise ValueError("sft_train_shuffle_seed must be non-negative.")
            self.sft_eval_tail_count = compute_sft_tail_eval_count(
                len(self.data),
                float(getattr(args, "sft_eval_tail_ratio", 0.0) or 0.0),
                int(getattr(args, "sft_eval_tail_docs", 0) or 0),
                require_train_docs=not int(getattr(args, "sft_eval_include_in_train", 0) or 0),
            )
            self.sft_eval_include_in_train = int(getattr(args, "sft_eval_include_in_train", 0) or 0)
            split_train_indices = None
            split_eval_indices = None
            if self.sft_train_shuffle and self.sft_eval_tail_count > 0:
                split_train_indices, split_eval_indices = compute_sft_shuffled_split_indices(
                    len(self.data),
                    self.sft_eval_tail_count,
                    eval_include_in_train=bool(self.sft_eval_include_in_train),
                    seed=self.sft_train_shuffle_seed,
                )
            if self.sft_split == "eval":
                if self.sft_eval_tail_count <= 0:
                    raise ValueError("SFT eval split requires sft_eval_tail_ratio > 0 or sft_eval_tail_docs > 0.")
                if split_eval_indices is not None:
                    self.sft_doc_start = 0
                    self.sft_doc_indices = split_eval_indices
                    self.sft_doc_count = len(split_eval_indices)
                else:
                    self.sft_doc_start = len(self.data) - self.sft_eval_tail_count
                    self.sft_doc_count = self.sft_eval_tail_count
            else:
                self.sft_doc_start = 0
                if self.sft_eval_include_in_train:
                    self.sft_doc_count = len(self.data)
                elif split_train_indices is not None:
                    self.sft_doc_indices = split_train_indices
                    self.sft_doc_count = len(split_train_indices)
                else:
                    self.sft_doc_count = len(self.data) - self.sft_eval_tail_count
            if self.sft_doc_count <= 0:
                raise ValueError(f"SFT {self.sft_split} split has no documents.")
            rank_zero_info(
                f"SFT mask data = {mask_file}; split={self.sft_split} "
                f"docs={self.sft_doc_count}/{len(self.data)} eval_tail={self.sft_eval_tail_count} "
                f"eval_include_in_train={self.sft_eval_include_in_train} "
                f"train_shuffle={self.sft_train_shuffle} shuffle_seed={self.sft_train_shuffle_seed}"
            )
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def _sft_epoch_permutation(self, epoch: int):
        epoch = int(epoch)
        if self._sft_shuffle_epoch != epoch or self._sft_shuffle_permutation is None:
            rng = np.random.default_rng(self.sft_train_shuffle_seed + epoch)
            self._sft_shuffle_permutation = rng.permutation(self.sft_doc_count)
            self._sft_shuffle_epoch = epoch
        return self._sft_shuffle_permutation

    def _sft_doc_index_from_sample(self, sample_index: int, epoch: int, epoch_sample_index: int | None = None) -> int:
        if self.sft_split == "train" and self.sft_train_shuffle:
            if epoch_sample_index is None:
                epoch_sample_index = sample_index - int(epoch) * self.samples_per_epoch
            offset = int(epoch_sample_index % self.sft_doc_count)
            offset = int(self._sft_epoch_permutation(epoch)[offset])
            return self._sft_doc_index_from_offset(offset)

        offset = int(sample_index % self.sft_doc_count)
        return self._sft_doc_index_from_offset(offset)

    def _sft_doc_index_from_offset(self, offset: int) -> int:
        if self.sft_doc_indices is not None:
            return int(self.sft_doc_indices[int(offset)])
        return self.sft_doc_start + offset

    def __len__(self):
        if self.data_type == "sft_binidx":
            if self.sft_split == "eval":
                eval_steps = int(getattr(self.args, "sft_eval_steps", 0) or 0)
                if eval_steps > 0:
                    return eval_steps * self.args.micro_bsz
                return self.sft_doc_count
            return self.args.epoch_steps * self.accumulate_grad_batches * self.args.micro_bsz
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        step_offset = self.step_offset
        if getattr(self, "data_type", getattr(args, "data_type", "binidx")) == "sft_binidx":
            step_offset *= self.accumulate_grad_batches
        logical_idx = idx + step_offset * args.micro_bsz
        epoch_sample_index = (logical_idx * world_size) + rank
        sample_index = epoch * self.samples_per_epoch + epoch_sample_index

        data_type = getattr(self, "data_type", getattr(args, "data_type", "binidx"))
        if data_type == "sft_binidx":
            doc_index = self._sft_doc_index_from_sample(sample_index, epoch, epoch_sample_index)
            token_ids = self.data[doc_index].astype(int)
            loss_mask = self.mask_data[doc_index].astype(int)
            if not np.isin(loss_mask, [0, 1]).all():
                raise ValueError(f"SFT loss mask contains values other than 0/1 in document {doc_index}.")

            req_len = args.ctx_len + 1
            if len(token_ids) > req_len:
                raise ValueError(
                    f"SFT document {doc_index} length {len(token_ids)} exceeds ctx_len + 1 ({req_len})."
                )
            if len(token_ids) < req_len:
                padding = req_len - len(token_ids)
                token_ids = np.concatenate(
                    [token_ids, np.full(padding, self.sft_pad_token_id, dtype=token_ids.dtype)]
                )
                loss_mask = np.concatenate([loss_mask, np.zeros(padding, dtype=loss_mask.dtype)])

            x = torch.tensor(token_ids[:-1], dtype=torch.long)
            y = torch.tensor(token_ids[1:], dtype=torch.long)
            target_mask = torch.tensor(loss_mask[1:], dtype=torch.float32)
            return x, y, target_mask

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + sample_index
        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y

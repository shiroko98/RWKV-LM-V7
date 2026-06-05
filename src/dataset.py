########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset

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
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_type = getattr(args, "data_type", "binidx")
        self.mask_data = None
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.real_epoch = getattr(args, "resume_epoch", args.epoch_begin)
        self.step_offset = getattr(args, "resume_step_offset", 0)

        if self.data_type == "binidx":
            assert self.samples_per_epoch == 40320
            dataset_slot = self.data_size // args.ctx_len
            assert is_prime(args.magic_prime)
            assert args.magic_prime % 3 == 2
            assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1
        elif self.data_type == "sft_binidx":
            if len(self.data) == 0:
                raise ValueError("SFT token dataset must contain at least one document.")
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
            rank_zero_info(f"SFT mask data = {mask_file}")
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        logical_idx = idx + self.step_offset * args.micro_bsz
        sample_index = epoch * self.samples_per_epoch + (logical_idx * world_size) + rank

        data_type = getattr(self, "data_type", getattr(args, "data_type", "binidx"))
        if data_type == "sft_binidx":
            doc_index = sample_index % len(self.data)
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

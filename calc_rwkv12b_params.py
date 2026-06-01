#!/usr/bin/env python3
import sys

import torch

REPO_DIR = "/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train"
MODEL_PTH = "/mnt/data/Codes/RWKV/RWKV-Scale/RWKV7-12B-scale/outputs/pruned/rwkv7-g1f-12b-56l-importance.pth"
DATA_FILE = "/mnt/data/Codes/RWKV/megatron_data_process/data/test_text_document"  # Do not include .bin/.idx
CTX_LEN = 32768

sys.path.append(REPO_DIR)
from src.binidx import MMapIndexedDataset


def is_prime(n: int) -> bool:
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


def load_ckpt(path: str):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return obj


ckpt = load_ckpt(MODEL_PTH)

emb_key = None
for k in ["emb.weight", "_forward_module.emb.weight", "head.weight", "_forward_module.head.weight"]:
    if k in ckpt:
        emb_key = k
        break

if emb_key is None:
    raise RuntimeError("Cannot find emb.weight/head.weight in checkpoint")

VOCAB_SIZE = int(ckpt[emb_key].shape[0])
CKPT_N_EMBD = int(ckpt[emb_key].shape[1])

layer_ids = set()
for k in ckpt.keys():
    kk = k
    if kk.startswith("_forward_module."):
        kk = kk[len("_forward_module.") :]
    if kk.startswith("blocks."):
        parts = kk.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            layer_ids.add(int(parts[1]))

if not layer_ids:
    raise RuntimeError("Cannot infer n_layer from checkpoint")

CKPT_N_LAYER = max(layer_ids) + 1

data = MMapIndexedDataset(DATA_FILE)
MY_EXIT_TOKENS = len(data._bin_buffer) // data._index._dtype_size

n_chunk = int(MY_EXIT_TOKENS // CTX_LEN) - 1
MAGIC_PRIME = None
for i in range(n_chunk, 0, -1):
    if i % 3 == 2 and is_prime(i):
        MAGIC_PRIME = i
        break

if MAGIC_PRIME is None:
    raise RuntimeError("Failed to compute magic_prime")

print("===== Copy these into your training script =====")
print(f'VOCAB_SIZE="{VOCAB_SIZE}"')
print(f'MY_EXIT_TOKENS="{MY_EXIT_TOKENS}"')
print(f'MAGIC_PRIME="{MAGIC_PRIME}"')
print("===== Sanity check =====")
print(f'CKPT_N_LAYER="{CKPT_N_LAYER}"')
print(f'CKPT_N_EMBD="{CKPT_N_EMBD}"')
print(f'CTX_LEN="{CTX_LEN}"')

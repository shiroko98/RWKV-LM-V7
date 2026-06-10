#!/usr/bin/env bash
set -euo pipefail

#######################################################################################################################
#
# Production-oriented RWKV7 G1F 13.3B SFT launcher for ctx_len=86016, one full pass.
#
# Expected dataset prefix:
#   DATA_FILE.bin / DATA_FILE.idx / DATA_FILE.mask.bin / DATA_FILE.mask.idx
#
# This wrapper only sets defaults, then delegates to run_13b_sft_zero3_offload.sh.
# Override any value by exporting it before the command.
#
#######################################################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CTX_LEN="${CTX_LEN:-86016}"
export LOAD_MODEL="${LOAD_MODEL:-/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth}"
export DATA_FILE="${DATA_FILE:-/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B}"
export PROJ_DIR="${PROJ_DIR:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-ctx86016-onepass}"

export N_NODE="${N_NODE:-1}"
export GPU_PER_NODE="${GPU_PER_NODE:-8}"
export MICRO_BSZ="${MICRO_BSZ:-1}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

export SFT_ONE_PASS="${SFT_ONE_PASS:-1}"
export EPOCH_STEPS="${EPOCH_STEPS:-1}"
export EPOCH_COUNT="${EPOCH_COUNT:-1}"

export STRATEGY="${STRATEGY:-deepspeed_stage_3_offload}"
export GRAD_CP="${GRAD_CP:-1}"
export DS_BUCKET_MB="${DS_BUCKET_MB:-64}"
export SFT_MASKED_CE_CHUNK="${SFT_MASKED_CE_CHUNK:-512}"

export LR_INIT="${LR_INIT:-5e-6}"
export LR_FINAL="${LR_FINAL:-5e-7}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-0}"
export LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-cosine}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

export EPOCH_SAVE="${EPOCH_SAVE:-1}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-0}"
export KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:-6}"

export WANDB_PROJECT="${WANDB_PROJECT:-RWKV-13B-SFT}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-16}"

exec bash "$SCRIPT_DIR/run_13b_sft_zero3_offload.sh"

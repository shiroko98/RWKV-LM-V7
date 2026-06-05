#!/usr/bin/env bash
set -euo pipefail

#######################################################################################################################
#
# RWKV7 G1F 13.3B SFT launcher for binidx + mask-sidecar data.
#
# Prepare data first. For example, for CTX_LEN=8192:
#   python data/make_sft_binidx.py /path/to/sft_jsonl_dir \
#     --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
#     --ctx-len 8192 \
#     --pack \
#     --num-workers 32 \
#     --shuffle
#
# The resulting training prefix must have:
#   DATA_FILE.bin / DATA_FILE.idx / DATA_FILE.mask.bin / DATA_FILE.mask.idx
#
#######################################################################################################################

MODEL_TYPE="${MODEL_TYPE:-x070}"

N_LAYER="${N_LAYER:-61}"
N_EMBD="${N_EMBD:-4096}"
DIM_FFN="${DIM_FFN:-16384}"
VOCAB_SIZE="${VOCAB_SIZE:-65536}"
HEAD_SIZE="${HEAD_SIZE:-64}"

D_DECAY_LORA="${D_DECAY_LORA:-192}"
D_AAA_LORA="${D_AAA_LORA:-192}"
D_MV_LORA="${D_MV_LORA:-128}"
D_GATE_LORA="${D_GATE_LORA:-384}"

CTX_LEN="${CTX_LEN:-8192}"
LOAD_MODEL="${LOAD_MODEL:-/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth}"
DATA_FILE="${DATA_FILE:-/mnt/data/datasets/sft_train_ctx8192}"
PROJ_DIR="${PROJ_DIR:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload}"

#######################################################################################################################
#
# SFT schedule.
#
# EPOCH_STEPS is optimizer steps per SFT epoch. If each packed document is one sample, a common starting point is:
#   ceil(num_documents / (NUM_NODES * GPU_PER_NODE * MICRO_BSZ))
# If EPOCH_STEPS * real_bsz is larger than the document count, the SFT dataset wraps around deterministically.
#
#######################################################################################################################

EPOCH_STEPS="${EPOCH_STEPS:-1000}"
EPOCH_COUNT="${EPOCH_COUNT:-1}"
EPOCH_SAVE="${EPOCH_SAVE:-1}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-0}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:-3}"

MICRO_BSZ="${MICRO_BSZ:-1}"
LR_INIT="${LR_INIT:-1e-5}"
LR_FINAL="${LR_FINAL:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
GRAD_CP="${GRAD_CP:-1}"
HEAD_CHUNK="${HEAD_CHUNK:-0}"
KERNEL="${KERNEL:-@rwkv3}"

N_NODE="${N_NODE:-1}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
STRATEGY="${STRATEGY:-deepspeed_stage_3_offload}"
DS_BUCKET_MB="${DS_BUCKET_MB:-64}"
MASTER_PORT="${MASTER_PORT:-29501}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/.torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-16}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT

if [[ ! -e "$LOAD_MODEL" ]]; then
  echo "Missing LOAD_MODEL: $LOAD_MODEL" >&2
  exit 1
fi

for suffix in .bin .idx .mask.bin .mask.idx; do
  if [[ ! -f "${DATA_FILE}${suffix}" ]]; then
    echo "Missing SFT dataset file: ${DATA_FILE}${suffix}" >&2
    exit 1
  fi
done

mkdir -p "$PROJ_DIR"
mkdir -p "$TORCH_EXTENSIONS_DIR"

EXTRA_ARGS=()
if [[ -n "${SFT_MASK_FILE:-}" ]]; then
  EXTRA_ARGS+=(--sft_mask_file "$SFT_MASK_FILE")
fi

python train.py --load_model "$LOAD_MODEL" --wandb "${WANDB_PROJECT:-}" --proj_dir "$PROJ_DIR" --my_testing "$MODEL_TYPE" \
 --ctx_len "$CTX_LEN" --train_stage 0 --epoch_steps "$EPOCH_STEPS" --epoch_count "$EPOCH_COUNT" --epoch_begin "${EPOCH_BEGIN:-0}" \
 --data_file "$DATA_FILE" --my_exit_tokens 0 --magic_prime 0 \
 --num_nodes "$N_NODE" --micro_bsz "$MICRO_BSZ" --n_layer "$N_LAYER" --n_embd "$N_EMBD" --dim_ffn "$DIM_FFN" --kernel "$KERNEL" \
 --lr_init "$LR_INIT" --lr_final "$LR_FINAL" --warmup_steps "$WARMUP_STEPS" --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
 --data_type "sft_binidx" --vocab_size "$VOCAB_SIZE" \
 --weight_decay "$WEIGHT_DECAY" --epoch_save "$EPOCH_SAVE" --save_every_n_steps "$SAVE_EVERY_N_STEPS" --keep_last_n_checkpoints "$KEEP_LAST_N_CHECKPOINTS" \
 --head_size "$HEAD_SIZE" --head_chunk "$HEAD_CHUNK" \
 --accelerator gpu --devices "$GPU_PER_NODE" --precision bf16 --strategy "$STRATEGY" --grad_cp "$GRAD_CP" --enable_progress_bar True --ds_bucket_mb "$DS_BUCKET_MB" --master_port "$MASTER_PORT" \
 --d_decay_lora "$D_DECAY_LORA" --d_aaa_lora "$D_AAA_LORA" --d_mv_lora "$D_MV_LORA" --d_gate_lora "$D_GATE_LORA" \
 "${EXTRA_ARGS[@]}"

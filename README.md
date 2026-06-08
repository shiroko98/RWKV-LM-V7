<div align="center">

# RWKV-LM-V7
[![English](https://img.shields.io/badge/README-English-blue.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-中文版本-red.svg)](./README_CN.md)

</div>

## Project Introduction

This project allows any researcher to start pre-training a fully aligned RWKV v7 model within 15 minutes. Of course, this does not include the time to download the data :)

All code is sourced from the original RWKV-LM project: https://github.com/BlinkDL/RWKV-LM

This repository is suitable for quickly reproducing small-scale RWKV v7 series models (e.g., 191M to 3B) on NVIDIA & AMD GPUs using either sample data or private data. We will focus on the following improvements next:

-   Provide template code for RWKV series models for tasks such as multimodal applications.
-   Provide cross-platform kernel implementations.
-   Provide a configurable RWKV Layer class.
-   Provide a high-performance PyTorch inference implementation.
-   Provide a cluster training framework and scripts suitable for models from 3B to 70B.

We love and give back to the open-source community and appreciate any implementations from it. If you find any issues in our code repository, including but not limited to code quality, code style, code interpretability, or numerical precision errors, you are welcome to [submit an issue](https://github.com/RWKV-Vibe/RWKV-LM-V7/issues/new).

> [!WARNING]
> Note: This is WIP (very likely correct, and more efficient). On the other hand, you can still use [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) as reference implementation.

## How to start?

### Prepare Environment

To prepare the environment, please use UV that is very fast and easy to use.
```
uv venv ./rwkv-lm --python 3.12
source ./rwkv-lm/bin/activate
```
Next, install the following dependencies. Please note that `pytorch-lightning` is fixed at version `1.9.5`. This is a specific requirement for this repository; do not upgrade this package.
```
uv pip install torch
uv pip install -r requirements.txt
```

### Download Data

```
wget -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```
### Start Training

1. Initialize an empty RWKV-7 model
```
sh ./demo-training-prepare.sh
```

2. Log in to your WandB account

3. Start training
```
sh ./demo-training-run.sh
```

## 13.3B Workflow

This repository also contains a practical 13.3B / DeepSpeed workflow for long-running training, clean shutdown, resume, checkpoint conversion, and inference.

### Start 13.3B Training

The main launcher script is [run_12b_zero3_offload.sh](/D:/codes/RWKV-LM-V7-12B-train/run_12b_zero3_offload.sh). Edit the model path, dataset path, context length, and the three dataset-dependent values first:

- `VOCAB_SIZE`
- `MY_EXIT_TOKENS`
- `MAGIC_PRIME`

Then start training:

```bash
bash run_12b_zero3_offload.sh
```

Useful training-side notes:

- This script currently uses `--strategy deepspeed_stage_3_offload`
- `--save_every_n_steps` and `--keep_last_n_checkpoints` are supported by `train.py`
- step checkpoints are named like `rwkv-step-200.pth`
- epoch checkpoints are named like `rwkv-10.pth`

### Stop Training Cleanly

Use [scripts/stop_rwkv_train.sh](/D:/codes/RWKV-LM-V7-12B-train/scripts/stop_rwkv_train.sh) instead of killing random worker PIDs.

Find the launcher PID:

```bash
pgrep -fo 'python .*train\.py'
```

Stop it cleanly:

```bash
bash scripts/stop_rwkv_train.sh "$(pgrep -fo 'python .*train\.py')"
```

The stop script sends:

1. `SIGINT`
2. `SIGTERM`
3. `SIGKILL`

This greatly reduces the noisy NCCL / TCPStore broken-pipe shutdown spam compared with force-killing individual workers.

### Resume Training

For DeepSpeed sharded checkpoints, resume by pointing `--load_model` at the checkpoint directory itself, for example:

```bash
--load_model /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-200.pth
```

`train.py` now detects DeepSpeed checkpoint directories automatically and passes them into Lightning via `ckpt_path`. This supports:

- `deepspeed_stage_1`
- `deepspeed_stage_2`
- `deepspeed_stage_2_offload`
- `deepspeed_stage_3`
- `deepspeed_stage_3_offload`

In practice, resuming usually just means:

1. edit `LOAD_MODEL` in [run_12b_zero3_offload.sh](/D:/codes/RWKV-LM-V7-12B-train/run_12b_zero3_offload.sh) to the checkpoint directory you want
2. run the same script again

Example:

```bash
bash run_12b_zero3_offload.sh
```

### Smoke-Test Resume Before a Long Run

If you want to verify that a real ZeRO checkpoint can resume correctly before committing to a long training run, use [scripts/regression_resume_deepspeed_checkpoint.py](/D:/codes/RWKV-LM-V7-12B-train/scripts/regression_resume_deepspeed_checkpoint.py):

```bash
python scripts/regression_resume_deepspeed_checkpoint.py \
  --checkpoint-path /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-200.pth \
  --log-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/resume-smoke-step200.log \
  --timeout-seconds 2400 \
  --steady-seconds 180 \
  -- \
  bash run_12b_zero3_offload.sh
```

This script:

- validates that the checkpoint path looks like a real DeepSpeed sharded checkpoint
- launches your normal training command
- waits for both resume markers and actual training progress
- stops the process group cleanly once resume looks healthy

### Convert a DeepSpeed Checkpoint to a Single `.pth`

Use [scripts/convert_deepspeed_checkpoint_to_pth.py](/D:/codes/RWKV-LM-V7-12B-train/scripts/convert_deepspeed_checkpoint_to_pth.py):

```bash
python scripts/convert_deepspeed_checkpoint_to_pth.py \
  --checkpoint-dir /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.pth \
  --output-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.bf16.pth \
  --dtype bf16
```

Optional summary file:

```bash
python scripts/convert_deepspeed_checkpoint_to_pth.py \
  --checkpoint-dir /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.pth \
  --output-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.bf16.pth \
  --dtype bf16 \
  --summary-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/rwkv-step-20-summary.txt
```

### Verify the Converted `.pth` Matches the ZeRO Checkpoint

Use [scripts/test_converted_checkpoint_equivalence.py](/D:/codes/RWKV-LM-V7-12B-train/scripts/test_converted_checkpoint_equivalence.py) to compare:

- reconstructed state_dict from the original ZeRO checkpoint
- converted single-file `.pth`
- forward logits on a real prompt

Important: always compare the same step against itself. Do not mix `rwkv-step-20.pth` with `rwkv-step-200.bf16.pth`.

Example:

```bash
python scripts/test_converted_checkpoint_equivalence.py \
  --checkpoint-dir /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.pth \
  --converted-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.bf16.pth \
  --dtype bf16 \
  --device cuda \
  --strict-forward \
  --summary-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/rwkv-step-20-equivalence.json
```

A successful run should end with:

```text
[equiv] state_dict match: ...
[equiv] forward check: ...
[equiv] PASS
```

If `max_abs_diff=0.0` and `topk_match=True`, the converted checkpoint matches exactly for both parameters and forward behavior.

### Run Inference on a Converted Single-File Checkpoint

Use [scripts/run_converted_rwkv_demo.py](/D:/codes/RWKV-LM-V7-12B-train/scripts/run_converted_rwkv_demo.py):

```bash
python scripts/run_converted_rwkv_demo.py \
  --model-path /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.bf16.pth \
  --vocab-path /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/data/tokenizer/rwkv_vocab_v20230424.txt \
  --device cuda \
  --dtype auto \
  --prompt "The Eiffel tower is in the city of" \
  --topk 10 \
  --max-new-tokens 32
```

Sampling example:

```bash
python scripts/run_converted_rwkv_demo.py \
  --model-path /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload/rwkv-step-20.bf16.pth \
  --vocab-path /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/data/tokenizer/rwkv_vocab_v20230424.txt \
  --device cuda \
  --dtype auto \
  --prompt "The Eiffel tower is in the city of" \
  --max-new-tokens 64 \
  --sample \
  --temperature 1.0 \
  --top-p 0.9
```

This demo automatically infers:

- `n_layer`
- `n_embd`
- `head_size`
- `D_DECAY_LORA`
- `D_AAA_LORA`
- `D_MV_LORA`
- `D_GATE_LORA`

from the checkpoint itself, so you do not need to hardcode 13.3B layout values by hand.

### Legacy `rwkv_v7_demo.py`

There is also a local [rwkv_v7_demo.py](/D:/codes/RWKV-LM-V7-12B-train/rwkv_v7_demo.py) file that was adapted for prompt continuation. It is more manual and keeps its own hardcoded paths and settings.

Use it only if you specifically want that standalone demo style. For practical converted-checkpoint inference, prefer `scripts/run_converted_rwkv_demo.py`.

## Detailed Explanation

This section contains explanations of model initialization, learning rates, and other details.

RWKV-7 uses initializations that are both theoretically designed with mathematical proof and empirically derived from training results to accelerate model convergence and improve performance.

### RWKV G1 model Lora Dim

|             | params | 0.1B | 0.4B | 1.5B | 2.9B | 7.2B | 13.3B |  
|-------------|--------|------|------|------|------|------|-------| 
|D_DECAY_LORA |   w    |  64  |  64  |  96  |  96  |  128 |  192  |
|D_AAA_LORA   |   a    |  64  |  64  |  96  |  96  |  128 |  192  |
|D_MV_LORA    |   v    |  32  |  32  |  64  |  64  |  96  |  128  |   
|D_GATE_LORA  |   g    |  128 |  128 |  256 |  320 |  480 |  384  |

### L2Warp

This type of penalty prevents the model from becoming overconfident, thereby mitigating precision loss in BF16.

### Weights and Initialization Example

Please pay close attention to the learning rate and related settings in the context.

```python
self.k_k = nn.Parameter(torch.zeros(1, 1, C)+0.71 - linear*0.1)
self.k_a = nn.Parameter(torch.zeros(1, 1, C)+1.02)
```

RWKV-7 weight example for 1.5B (L24-D2048, vocab 65536):

| name                | shape         | comment      | initialization  |
|---------------------|---------------|--------------|-----------------|
| emb.weight          | [65536, 2048] | wdecay       | see code        |
| blocks.0.ln0.weight | [2048]        | for layer 0  | 1               |
| blocks.0.ln0.bias   | [2048]        | for layer 0  | 0               |
|                     |               |              |                 |
| blocks.*.ln1.weight | [2048]        |              | 1               |
| blocks.*.ln1.bias   | [2048]        |              | 0               |
| blocks.*.att.x_r    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_w    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_k    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_v    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_a    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_g    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.w0     | [1, 1, 2048]  | lr 2x        | see code        |
| blocks.*.att.w1     | [2048, 96]    |              | 0               |
| blocks.*.att.w2     | [96, 2048]    |              | see code        |
| blocks.*.att.a0     | [1, 1, 2048]  |              | 0               |
| blocks.*.att.a1     | [2048, 96]    |              | 0               |
| blocks.*.att.a2     | [96, 2048]    |              | see code        |
| blocks.*.att.v0     | [1, 1, 2048]  | for layer 1+ | 1               |
| blocks.*.att.v1                | [2048, 64]   | for layer 1+ | 0         |
| blocks.*.att.v2                | [64, 2048]   | for layer 1+ | see code  |
| blocks.*.att.g1                | [2048, 256]  |              | 0         |
| blocks.*.att.g2                | [256, 2048]  |              | see code  |
| blocks.*.att.k_k               | [1, 1, 2048] |              | 1         |
| blocks.*.att.k_a               | [1, 1, 2048] |              | 1         |
| blocks.*.att.r_k               | [32, 64]     |              | 0         |
| blocks.*.att.receptance.weight | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.key.weight        | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.value.weight      | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.output.weight     | [2048, 2048] | wdecay       | 0         |
| blocks.*.att.ln_x.weight       | [2048]       |              | see code  |
| blocks.*.att.ln_x.bias         | [2048]       |              | 0         |
|                                |              |              |           |
| blocks.*.ln2.weight            | [2048]       |              | 1         |
| blocks.*.ln2.bias              | [2048]       |              | 0         |
| blocks.*.ffn.x_k               | [1, 1, 2048] |              | see code  |
| blocks.*.ffn.key.weight        | [8192, 2048] | wdecay       | see code  |
| blocks.*.ffn.value.weight      | [2048, 8192] | wdecay       | 0         |
|                                |              |              |           |
| ln_out.weight | [2048]        |        | 1         |
| ln_out.bias   | [2048]        |        | 0         |
| head.weight   | [65536, 2048] | wdecay | see code  |

## Check Result
your `out/....../train_log.txt` should have losses similar to:
```
0 4.875856 131.0863 0.00059975 2025-04-24 02:23:42.481256 0
1 4.028621 56.1834 0.00059899 2025-04-24 02:28:16.674463 1
2 3.801625 44.7739 0.00059773 2025-04-24 02:32:51.059568 2
3 3.663070 38.9808 0.00059597 2025-04-24 02:37:25.409892 3
4 3.578974 35.8368 0.00059371 2025-04-24 02:41:59.711315 4
5 3.510906 33.4786 0.00059096 2025-04-24 02:46:33.990839 5
6 3.462345 31.8917 0.00058771 2025-04-24 02:51:08.378331 6
7 3.412196 30.3318 0.00058399 2025-04-24 02:55:42.927474 7
8 3.376724 29.2747 0.00057978 2025-04-24 03:00:17.504665 8
9 3.336911 28.1321 0.00057511 2025-04-24 03:04:52.006063 9
10 3.313411 27.4787 0.00056999 2025-04-24 03:09:27.563336 10
11 3.295895 27.0016 0.00056441 2025-04-24 03:14:01.786079 11
```

## Processing training data

### Convert jsonl to binidx format

Use `data/make_data.py` script to convert your training data from `.jsonl` format to `binidx` format.

```
python data/make_data.py [input_file] [n_epoch] [ctx_len]
# example:
cd data/
python make_data.py demo.jsonl 3 4096
```

This command will:

- shuffle & duplicate demo.jsonl (for 3 epochs)
- load jsonl and tokenize
- save as demo.bin & demo.idx
- compute "magic_prime" for ctxlen 4096

Assume your source jsonl is:

- {"text":"aa"}
- {"text":"bb"}
- {"text":"cc"}
- {"text":"dd"}

The final binidx will be like (here "/" means end_of_doc, which is actually token [0]): `bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/`

> [!WARNING]
> make_data.py will be very slow for large jsonl,check [json2binidx_tool](https://github.com/Abel2076/json2binidx_tool) if you need to process large jsonl.

### Convert SFT messages jsonl to binidx + loss mask

SFT preprocessing uses `data/make_sft_binidx.py`. The input is one JSON object per line. Each object should contain `messages`, and may optionally contain `tools`. `messages` follows a chat-style structure with roles such as `system`, `user`, `assistant`, and `tool`. Assistant tool-call arguments are normalized before rendering: JSON strings are parsed into structured arguments, while existing XML-style parameter fragments are preserved.

Example:

```bash
python data/make_sft_binidx.py data/sft_part_000.jsonl data/sft_part_001.jsonl \
  --out-prefix data/sft_train \
  --vocab rwkv_vocab_v20260603.txt \
  --chat-template data/SFT/sample/chat_template.jinja \
  --ctx-len 4096 \
  --pack \
  --num-workers 8 \
  --shuffle
```

You can also pass a directory containing JSONL shards. A directory input recursively expands to all nested `*.jsonl` files in sorted path order:

```bash
python data/make_sft_binidx.py data/sft_shards \
  --output-prefix data/sft_train \
  --ctx-len 4096 \
  --pack \
  --num-workers 8
```

To keep the original order across input files and epochs, disable shuffling:

```bash
python data/make_sft_binidx.py data/sft_part_000.jsonl data/sft_part_001.jsonl \
  --out-prefix data/sft_train_ordered \
  --no-shuffle
```

Use `--out-prefix` or its alias `--output-prefix` to choose the output binidx prefix. For example, `--out-prefix data/sft_train` writes `data/sft_train.bin`, `data/sft_train.idx`, `data/sft_train.mask.bin`, and `data/sft_train.mask.idx`. When passing more than one positional input path, the prefix is required because there is no single source name to infer it from. A single file defaults to that file name without `.jsonl`; a single directory defaults to the directory path as the prefix.

`--num-workers` is used in two places. On the default path, each JSONL file is one read task, so multiple files can be read concurrently; one large file is not split across workers at the read stage. `--pack-strategy best-fit-decreasing` reads and packs JSONL shards in bounded groups. `--pack-shard-group-size` defaults to `1`, which preserves the previous one-shard-at-a-time memory profile; raising it lets each group read multiple JSONL shards concurrently and best-fit pack across that group. Groups themselves are processed serially: group 2 starts only after group 1 has been read, tokenized, packed, and appended to the output builders. Inside a group, JSONL read concurrency is `min(group_size, num_workers, files_in_group)`. For example, `--pack-shard-group-size 8 --num-workers 32` reads at most 8 JSONL files at the same time; `--pack-shard-group-size 64 --num-workers 32` reads at most 32 files at once and queues the remaining files. Template rendering and tokenization inside each group are still parallelized across source samples, up to `num_workers`. Output writes remain single-threaded append operations into one token binidx plus one mask sidecar. Output order remains deterministic, so the same inputs, `--seed`, shuffle setting, and group size produce reproducible datasets. Text files are read as UTF-8; JSONL also accepts a UTF-8 BOM. Chinese and other multi-byte text are mapped through UTF-8 byte spans, so mask projection does not lose non-ASCII content.

Example best-fit commands:

```bash
# Recommended bounded multi-file packing:
# groups are serial; each group has at most 8 JSONL files; reads use at most 8 workers here.
python data/make_sft_binidx.py /mnt/data/datasets/sft_jsonl \
  --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
  --ctx-len 8192 \
  --pack \
  --pack-strategy best-fit-decreasing \
  --pack-shard-group-size 8 \
  --num-workers 32 \
  --shuffle

# Larger group, same worker cap:
# each group has up to 64 JSONL files; at most 32 files are read concurrently, the rest queue.
python data/make_sft_binidx.py /mnt/data/datasets/sft_jsonl \
  --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
  --ctx-len 8192 \
  --pack \
  --pack-strategy best-fit-decreasing \
  --pack-shard-group-size 64 \
  --num-workers 32 \
  --shuffle
```

The high-level flow is:

1. Read one or more UTF-8 JSONL files, skip empty lines, and keep source path plus line number for error reporting.
2. Expand directory inputs recursively to all sorted `*.jsonl` files.
3. Repeat the source samples by `--n-epoch`. This is offline duplication before writing the dataset: `--n-epoch 3` writes each source sample three times into the produced binidx. The default is `--n-epoch 1`, so samples are not duplicated unless you explicitly raise it. By default each repeated pass is deterministically shuffled with `--seed`; `--no-shuffle` keeps input order for every pass.
4. Load the authoritative SFT chat template from `data/SFT/sample/chat_template.jinja`. The root template is not used by this preprocessing path.
5. Normalize tool-call arguments, then render each sample twice with the same Jinja template: one prefix render up to the final assistant turn for the context boundary, and one full render for the actual training text.
6. Normalize the final assistant turn so it always contains think tags. Existing think content is preserved; missing think content receives an empty think block before the visible reply. Historical assistant turns, system text, user text, and tool outputs are context only.
7. Derive the loss mask from the final assistant trainable suffix: everything before the final assistant content boundary is `0`. Real think content that comes from the sample is trainable, but the automatically added empty think block for no-think samples is context only and remains `0`; the visible reply, final tool calls, assistant ending segment, and real sample ending segment are `1`.
8. Tokenize the final text once. The code records each token's UTF-8 byte span, maps it back to character spans, and projects the character-level trainable region into a token-level mask. This handles Chinese, multi-byte symbols, and special fragments through the same path.
9. Without `--pack`, `--pad`, `--pack-length`, or `--pad-length`, each repeated source sample becomes one variable-length binidx document and one same-length mask document; no padding is added. Each independent document still ends with the real `EOD_TOKEN`, and that EOD is trainable. `--ctx-len`, `--pack-length`, and `--pad-length` are token counts, not character counts. The recommended form is `--ctx-len N --pack` or `--ctx-len N --pad`; the actual preprocessing length is `N + 1` to match next-token labels at training time. With `--pack` / `--pack-length`, samples are packed without splitting a sample. The default `ordered` strategy preserves input order: multiple complete samples may share one fixed-length document, the separator newline between samples is masked out, and if the next complete sample does not fit, the current document is right-padded and a new one starts. `--pack-strategy best-fit-decreasing` sorts samples by token length and uses a best-fit approximation to reduce padding; it still never splits a sample, but it does reorder samples. This strategy is applied independently inside each JSONL shard group. `--pack-shard-group-size 1` means one JSONL per group; larger values allow cross-file best-fit within a bounded batch, improving packing efficiency while avoiding all-data-in-memory behavior. All processed groups are appended into one final `PREFIX` binidx + mask output. With `--pad` / `--pad-length`, packing stays disabled and each source sample is padded independently. Overlong samples are filtered after tokenization and before packing/padding according to the target token length; they are dropped instead of stopping the whole build.
10. The output is the token dataset plus a mask sidecar: `PREFIX.bin`, `PREFIX.idx`, `PREFIX.mask.bin`, and `PREFIX.mask.idx`. SFT training reads tokens from the main dataset and loss participation from the sidecar mask.

Main parameters:

- `--chat-template`: SFT render template path. Defaults to `data/SFT/sample/chat_template.jinja`.
- `--vocab`: tokenizer vocab. Defaults to `rwkv_vocab_v20260603.txt`.
- `--out-prefix` / `--output-prefix`: output binidx prefix. The four output files are derived from this prefix.
- `--n-epoch`: offline data repetition count. The default is `1`; values greater than `1` duplicate samples in the produced dataset.
- `--seed`: random seed for shuffling; it does not affect order when shuffle is disabled.
- `--shuffle` / `--no-shuffle`: whether to shuffle samples inside each epoch. Default is enabled.
- `--num-workers`: worker count for concurrent reading, rendering, and tokenization. Default is `1`.
- `--ctx-len`: training context length in tokens. With `--pack` or `--pad`, preprocessing uses `ctx_len + 1`.
- `--pack`: enable ordered sample-preserving packing at `ctx_len + 1`. Disabled by default.
- `--pack-strategy`: packing strategy. Defaults to `ordered`; `best-fit-decreasing` reorders by length inside each JSONL shard group, uses a best-fit approximation to reduce padding, and appends all groups into one output dataset.
- `--pack-shard-group-size`: number of JSONL shards per best-fit group. Defaults to `1`; larger values enable concurrent multi-JSONL reads and cross-file packing inside each bounded group.
- `--pad`: enable per-sample padding at `ctx_len + 1`. Disabled by default and mutually exclusive with `--pack`.
- `--pack-length`: legacy explicit fixed-length packing target in tokens.
- `--pad-length`: legacy explicit fixed-length per-sample padding target in tokens. Mutually exclusive with `--pack-length`.
- `--current-date`, `--current-location`: override or inject date and location fields in the system message.

### Train with SFT binidx data

Use `--data_type sft_binidx` when training on the SFT preprocessing output. `--data_file` is the binidx prefix without `.bin` or `.idx`; the trainer automatically loads `DATA_FILE.mask` unless `--sft_mask_file` is provided.

For SFT, `--epoch_steps` and `--epoch_count` are user-controlled. `epoch_steps` is the number of optimizer steps per epoch, and `epoch_count` is the number of epochs to run. This is different from pretraining `binidx`, where `train.py` keeps the historical magic-prime schedule.

```bash
python train.py \
  --load_model model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --proj_dir out/sft-0.4b \
  --data_file data/sft_train \
  --data_type sft_binidx \
  --ctx_len 4096 \
  --epoch_steps 1000 \
  --epoch_count 1 \
  --micro_bsz 1 \
  --vocab_size 65536 \
  --n_layer 24 \
  --n_embd 1024 \
  --dim_ffn 4096 \
  --head_size 64 \
  --d_decay_lora 64 \
  --d_aaa_lora 64 \
  --d_mv_lora 32 \
  --d_gate_lora 128 \
  --my_testing x070 \
  --lr_init 1e-5 \
  --lr_final 1e-5 \
  --warmup_steps 10 \
  --weight_decay 0 \
  --accelerator gpu \
  --devices 1 \
  --precision bf16 \
  --strategy deepspeed_stage_2 \
  --grad_cp 1
```

In this generic example, `--accelerator gpu` tells Lightning to train on CUDA GPUs, and `--devices 1` means one GPU in the current node. For multi-GPU DeepSpeed on one node, set `--devices` to the GPU count, for example `--devices 8`; `train.py` will automatically relaunch itself with `torchrun` when `strategy` contains `deepspeed`, `num_nodes=1`, and `devices > 1`. The global batch used by SFT scheduling is `real_bsz = num_nodes * devices * micro_bsz`. `--my_exit_tokens` is intentionally omitted for SFT because SFT stops by `--epoch_count`; `my_exit_tokens` is part of the pretraining token-limit schedule.

Training samples use next-token labels, so the dataloader needs `ctx_len + 1` token ids per SFT document. A shorter document is padded in memory with `--sft_pad_token_id` and mask `0`; a longer document raises an error. For predictable fixed-length training, build data with `--ctx-len CTX_LEN --pack` or `--ctx-len CTX_LEN --pad`; preprocessing writes `CTX_LEN + 1` token documents, then train with `--ctx_len CTX_LEN`. For RWKV7 x070, keep `ctx_len` divisible by 16.

The 0.4B checkpoint listed in the example is `L24-D1024` with `dim_ffn=4096`, `vocab_size=65536`, `head_size=64`, and RWKV7 G1 LoRA dimensions `64/64/32/128`. If you use another checkpoint, read its architecture text and keep these shape parameters aligned with the checkpoint.

SFT masked-training implementation:

1. Offline preprocessing writes two aligned binidx datasets. `PREFIX.bin/.idx` stores token ids, while `PREFIX.mask.bin/.idx` stores the same-length `0/1` loss mask.
2. `train.py` enters SFT mode with `--data_type sft_binidx`. This mode does not use the pretraining magic-prime schedule. It preserves user-provided `--epoch_steps` and `--epoch_count`, and sets Lightning `max_epochs` to `epoch_count`.
3. `src/dataset.py::MyDataset` loads the token prefix from `--data_file` and the mask prefix from `--data_file.mask` by default. `--sft_mask_file` can override the mask prefix. Initialization validates matching document counts and identical per-document sizes.
4. Each SFT document may contain at most `ctx_len + 1` tokens. Short documents are padded in memory with `--sft_pad_token_id` and mask `0`; long documents raise an error instead of being silently truncated.
5. The dataset returns `(x, y, loss_mask)`: `x = token_ids[:-1]`, `y = token_ids[1:]`, and `loss_mask = raw_mask[1:]`. The mask is shifted so it marks whether each next-token target contributes to loss.
6. `src/model.py::training_step` dispatches by batch shape. Pretraining batches `(x, y)` keep the existing fused CE path. SFT batches `(x, y, loss_mask)` use `src/sft_loss.py::masked_cross_entropy`.
7. `masked_cross_entropy` computes per-token CE, averages only positions with `loss_mask=1`, and returns a differentiable zero loss if the mask is empty.

Validation coverage:

- Default unit tests in `tests/test_sft_training.py` cover sidecar loading, mask shifting, padding, invalid masks, too-long documents, SFT epoch scheduling, and masked CE math.
- `tests/test_sft_binidx.py` covers authoritative Jinja rendering, think/no-think rules, Chinese UTF-8 spans, tool calls, packing, padding, recursive directory input, concurrent JSONL loading, and CLI parsing.
- `RWKV_RUN_CUDA_SFT_SMOKE=1` loads a real RWKV7 checkpoint, builds a tiny SFT binidx dataset, and runs CUDA forward/backward with masked SFT loss.
- `RWKV_RUN_TRAIN_PY_SFT_SMOKE=1` launches `train.py` for one SFT step and covers Lightning, DeepSpeed, optimizer, and multi-card torchrun.
- `RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1` saves `rwkv-step-1.pth` and resumes from it, covering SFT checkpoint resume and DeepSpeed sharded checkpoint loading.

Current validation results:

- Local default regression: `115 passed, 3 skipped`.
- SFT training targeted coverage: `src.dataset`, `src.sft_loss`, and testable `train.py` helper surface are `100%`.
- SFT preprocessing coverage: `src.sft_binidx`, `data.make_sft_binidx`, and `data.tokenizer.rwkv_tokenizer` total `99%`.
- Server 8xH800:
  - `RWKV_RUN_CUDA_SFT_SMOKE=1` -> `3 passed, 2 skipped`.
  - `RWKV_RUN_TRAIN_PY_SFT_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_2` -> `3 passed, 2 skipped`.
  - `RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_3_offload` -> `3 passed, 2 skipped`.
  - `RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_3_offload` -> `1 passed in 284.01s`.

13.3B SFT launcher:

The complete 13.3B example is [run_13b_sft_zero3_offload.sh](/D:/codes/RWKV-LM-V7-12B-train/run_13b_sft_zero3_offload.sh). It is derived from `model/rwkv7-g1f-13.3b.txt`: `n_layer=61`, `n_embd=4096`, `dim_ffn=16384`, `vocab_size=65536`, `head_size=64`, and LoRA dimensions `192/192/128/384`. It defaults to 8xH800, `deepspeed_stage_3_offload`, activation checkpointing enabled, and SFT binidx+mask data.

Step 1: prepare fixed-length SFT data. `--ctx-len 8192 --pack` writes `8193` tokens per document, because training uses next-token labels:

```bash
python data/make_sft_binidx.py /mnt/data/datasets/sft_jsonl \
  --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
  --ctx-len 8192 \
  --pack \
  --pack-strategy best-fit-decreasing \
  --pack-shard-group-size 8 \
  --num-workers 32 \
  --shuffle
```

Step 2: calculate `EPOCH_STEPS` for one full pass over the produced SFT documents:

```bash
DATA_FILE=/mnt/data/datasets/sft_train_ctx8192 \
NUM_NODES=1 \
DEVICES=8 \
MICRO_BSZ=1 \
N_PASS=1 \
python - <<'PY'
import math, os
from src.binidx import MMapIndexedDataset

docs = len(MMapIndexedDataset(os.environ["DATA_FILE"]))
real_bsz = int(os.environ["NUM_NODES"]) * int(os.environ["DEVICES"]) * int(os.environ["MICRO_BSZ"])
epoch_steps = math.ceil(docs / real_bsz)

print(f"documents={docs}")
print(f"real_bsz={real_bsz}")
print(f"EPOCH_STEPS={epoch_steps}")
print(f"EPOCH_COUNT={int(os.environ['N_PASS'])}")
print(f"samples_per_epoch={epoch_steps * real_bsz}")
print(f"extra_repeated_per_epoch={epoch_steps * real_bsz - docs}")
PY
```

Step 3: launch 13.3B SFT on 8 H800 GPUs:

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth \
DATA_FILE=/mnt/data/datasets/sft_train_ctx8192 \
PROJ_DIR=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload \
CTX_LEN=8192 \
N_NODE=1 \
GPU_PER_NODE=8 \
MICRO_BSZ=1 \
EPOCH_STEPS=12500 \
EPOCH_COUNT=1 \
STRATEGY=deepspeed_stage_3_offload \
GRAD_CP=1 \
LR_INIT=1e-5 \
LR_FINAL=1e-6 \
WARMUP_STEPS=10 \
EPOCH_SAVE=1 \
SAVE_EVERY_N_STEPS=0 \
KEEP_LAST_N_CHECKPOINTS=3 \
WANDB_PROJECT=RWKV-13B-SFT \
bash run_13b_sft_zero3_offload.sh
```

Key parameters:

- `LOAD_MODEL`: initial 13.3B checkpoint, or a saved `rwkv-step-N.pth` / `rwkv-N.pth` checkpoint for resume. DeepSpeed checkpoint directories are supported when the strategy is DeepSpeed.
- `DATA_FILE`: SFT binidx prefix, without `.bin` or `.idx`. The script expects `DATA_FILE.bin`, `DATA_FILE.idx`, `DATA_FILE.mask.bin`, and `DATA_FILE.mask.idx`.
- `CTX_LEN`: training context length. It must match the preprocessing `--ctx-len`; the binidx documents should contain `CTX_LEN + 1` tokens.
- `N_NODE`, `GPU_PER_NODE`, `MICRO_BSZ`: define current real batch size: `real_bsz = N_NODE * GPU_PER_NODE * MICRO_BSZ`.
- `EPOCH_STEPS`: optimizer steps per SFT epoch. For one full pass, use `ceil(num_sft_documents / real_bsz)`.
- `EPOCH_COUNT`: number of SFT passes. For `N` passes over the SFT data, keep `EPOCH_STEPS` from the one-pass formula and set `EPOCH_COUNT=N`.
- `GRAD_CP`: activation checkpointing. `1` enables block-level checkpointing to save VRAM; `0` disables it and is faster if memory allows.
- Gradient accumulation: not wired in this script/code path yet. Do not multiply `real_bsz` by an accumulation factor.
- `STRATEGY`: defaults to `deepspeed_stage_3_offload` for lower VRAM. Use `deepspeed_stage_3` for pure ZeRO-3 if memory allows.
- `LR_INIT`, `LR_FINAL`, `WARMUP_STEPS`, `WEIGHT_DECAY`: SFT learning-rate schedule and regularization.
- `EPOCH_SAVE`, `SAVE_EVERY_N_STEPS`, `KEEP_LAST_N_CHECKPOINTS`: checkpoint cadence and retention.
- `PROJ_DIR`: output directory for logs and checkpoints.
- `WANDB_PROJECT`: empty disables wandb; a non-empty value enables logging under that project.

Optional CUDA smoke tests are available for server validation:

```bash
RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_CUDA_SFT_SMOKE=1 \
pytest -q tests/test_sft_cuda_smoke.py

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_SMOKE=1 \
pytest -q tests/test_sft_cuda_smoke.py

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1 \
pytest -q tests/test_sft_cuda_smoke.py

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_checkpoint_converts_to_pth
```

The first command runs an in-process CUDA forward/backward on SFT masked loss. The second command launches `train.py` for one SFT step and also validates the Lightning/DeepSpeed/optimizer path. The third command saves `rwkv-step-1.pth` and resumes from it, covering SFT checkpoint resume and DeepSpeed sharded checkpoint loading when a DeepSpeed strategy is used. The fourth command creates a tiny SFT DeepSpeed checkpoint, converts the sharded checkpoint directory to a single `.pth`, reloads it, and compares it against the reconstructed ZeRO checkpoint. On multi-GPU servers, add `RWKV_SFT_SMOKE_DEVICES=8`; `train.py` will relaunch with torchrun for multi-card DeepSpeed. You can also set `RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3` or `deepspeed_stage_3_offload` to validate a different sharding mode.

To test pth merge for an existing checkpoint directory instead of training a tiny one first:

```bash
RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1 \
RWKV_SFT_MERGE_CHECKPOINT_DIR=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1.pth \
RWKV_SFT_MERGE_OUTPUT_FILE=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1.bf16.pth \
RWKV_SFT_MERGE_STRICT_FORWARD=1 \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_checkpoint_converts_to_pth
```

### Compute magic_prime for specified binidx dataset

The `data/compute_magic_prime.py` script computes the correct values of `--my_exit_tokens` and `--magic_prime` for a specified binidx dataset and context length (ctx_len).

- change the `DATA_NAME` and `CTX_LEN` in the `data/compute_magic_prime.py` for your training dataset and context length
- run the script to get the correct values of `--my_exit_tokens` and `--magic_prime`

```
cd data/
python compute_magic_prime.py
```

output will be like:

```
### Loading /home/rwkv/RWKV-LM-V7/data/demo

### /home/rwkv/RWKV-LM-V7/data/demo.bin/idx has 200499 tokens, 546 items. Dtype <class 'numpy.uint16'>

### magic_prime = 47 (for ctxlen 4096)

--my_exit_tokens 200499 --magic_prime 47 --ctx_len 4096
```

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
  --pack-length 4096 \
  --num-workers 8 \
  --shuffle
```

To keep the original order across input files and epochs, disable shuffling:

```bash
python data/make_sft_binidx.py data/sft_part_000.jsonl data/sft_part_001.jsonl \
  --out-prefix data/sft_train_ordered \
  --no-shuffle
```

When passing more than one input JSONL, `--out-prefix` is required because there is no single source filename from which to infer the output prefix. `--num-workers` reads multiple JSONL files concurrently and also parallelizes template rendering plus tokenization. Output order remains deterministic, so the same inputs, `--seed`, and shuffle setting produce reproducible datasets. Text files are read as UTF-8; JSONL also accepts a UTF-8 BOM. Chinese and other multi-byte text are mapped through UTF-8 byte spans, so mask projection does not lose non-ASCII content.

The high-level flow is:

1. Read one or more UTF-8 JSONL files, skip empty lines, and keep source path plus line number for error reporting.
2. Repeat the source samples by `--n-epoch`. By default each epoch is deterministically shuffled with `--seed`; `--no-shuffle` keeps input order.
3. Load the authoritative SFT chat template from `data/SFT/sample/chat_template.jinja`. The root template is not used by this preprocessing path.
4. Normalize tool-call arguments, then render each sample twice with the same Jinja template: one prefix render up to the final assistant turn for the context boundary, and one full render for the actual training text.
5. Normalize the final assistant turn so it always contains think tags. Existing think content is preserved; missing think content receives an empty think block before the visible reply. Historical assistant turns, system text, user text, and tool outputs are context only.
6. Derive the loss mask from the final assistant trainable suffix: everything before the final assistant content boundary is `0`; the final assistant think block, visible reply, final tool calls, assistant ending segment, and real sample ending segment are `1`.
7. Tokenize the final text once. The code records each token's UTF-8 byte span, maps it back to character spans, and projects the character-level trainable region into a token-level mask. This handles Chinese, multi-byte symbols, and special fragments through the same path.
8. Without packing, each source sample becomes one binidx document and one same-length mask document. With `--pack-length`, real samples are concatenated into fixed-length documents; the separator newline between real samples and tail padding are both masked out.
9. The output is the token dataset plus a mask sidecar: `PREFIX.bin`, `PREFIX.idx`, `PREFIX.mask.bin`, and `PREFIX.mask.idx`. The future SFT dataloader should read tokens from the main dataset and loss participation from the sidecar mask.

Main parameters:

- `--chat-template`: SFT render template path. Defaults to `data/SFT/sample/chat_template.jinja`.
- `--vocab`: tokenizer vocab. Defaults to `rwkv_vocab_v20260603.txt`.
- `--n-epoch`: offline data repetition count.
- `--seed`: random seed for shuffling; it does not affect order when shuffle is disabled.
- `--shuffle` / `--no-shuffle`: whether to shuffle samples inside each epoch. Default is enabled.
- `--num-workers`: worker count for concurrent reading, rendering, and tokenization. Default is `1`.
- `--pack-length`: fixed-length packing target. If omitted, each source sample remains one document.
- `--current-date`, `--current-location`: override or inject date and location fields in the system message.

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

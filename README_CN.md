
<div align="center">

# RWKV-LM-V7
[![English](https://img.shields.io/badge/README-English-blue.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-中文版本-red.svg)](./README_CN.md)

</div>

## 项目介绍
让任何研究者在 15 分钟内开始预训练一个完全对齐的 RWKV v7 模型。当然不包括下载数据 :) 。

所有的代码来源于原始 RWKV-LM 项目：https://github.com/BlinkDL/RWKV-LM

此仓库适合快速的在Nvidia和AMD显卡上使用样例数据或私有数据小规模的复现 RWKV v7 系列模型，例如 191M ~ 3B 等大小，我们接下来会重点改善：

- 提供 RWKV 系列模型在多模态等任务的模板代码
- 提供跨平台的内核实现
- 提供可配置的 RWKV Layer 类
- 提供高性能的 Pytorch 推理实现
- 提供适合 3 ~ 70B 的集群训练框架及脚本

我们热爱并回馈开源社区，感谢任何开源社区的实现。如果您发现我们的代码仓库有包含但不限于：代码质量，代码风格，代码可解释性，数值精度误差的问题，欢迎[提交 issue](https://github.com/RWKV-Vibe/RWKV-LM-V7/issues/new)。

> [!WARNING]
> Note: 整个仓库仍处于 WIP 阶段（与基线相比，我们改进了融合算子的使用，计算更高效）。如果您有所顾虑，则可以使用 [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) 作为参考实现。

## 如何开始？

### 准备环境

环境准备，请使用 UV 嘎嘎快而且非常方便，用过的都说好 😉
```
uv venv ./rwkv-lm --python 3.12
source ./rwkv-lm/bin/activate
```
随后安装下列依赖，注意 `pytorch-lightning` 固定使用了 `1.9.5` 版本，此为本仓库特性，请不要升级此依赖包。
```
uv pip install torch
uv pip install -r requirements.txt
```

### 下载数据

```
wget -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

### 开始训练

1. 初始化空 RWKV-7 模型
```
sh ./demo-training-prepare.sh
```

2. 登录 WandB 账号

3. 开始训练
```
sh ./demo-training-run.sh
```

## 详细解释

此章节包含模型初始化、学习率及细节解释。

RWKV-7 使用了包含经过设计和数学论证的初始化和基于训练结果分析的初始化，加速模型收敛及其性能。

### RWKV G1 模型 Lora 维度

|             | params | 0.1B | 0.4B | 1.5B | 2.9B | 7.2B | 13.3B |  
|-------------|--------|------|------|------|------|------|-------| 
|D_DECAY_LORA |   w    |  64  |  64  |  96  |  96  |  128 |  192  |
|D_AAA_LORA   |   a    |  64  |  64  |  96  |  96  |  128 |  192  |
|D_MV_LORA    |   v    |  32  |  32  |  64  |  64  |  96  |  128  |   
|D_GATE_LORA  |   g    |  128 |  128 |  256 |  320 |  480 |  384  |

### L2Warp
此类惩罚模型，避免模型过度自信，从而缓解 BF16 中间的精度损失。

### 权重及其初始化样例

请严格注意上下文学习率等相关设置：
```
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

## 检查结果

在 `out/....../train_log.txt` 路径下，您的损失应该非常接近：

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

## 处理训练数据

### 将 jsonl 转换为 binidx 格式

使用 `data/make_data.py` 脚本将你的训练数据从 `.jsonl` 格式转换为 `binidx` 格式。

```
python make_data.py [输入文件] [重复轮数] [上下文长度]
# 示例：
cd data/
python make_data.py demo.jsonl 3 4096
```

`python make_data.py demo.jsonl 3 4096` 命令将会：

- 对 demo.jsonl 进行打乱并复制 3 轮
- 加载 jsonl 并进行 tokenize
- 保存为 demo.bin 和 demo.idx
- 计算 ctxlen=4096 时的 `magic_prime`

假设你的源 jsonl 如下：

- {"text":"aa"}
- {"text":"bb"}
- {"text":"cc"}
- {"text":"dd"}

最终的 binidx 会像这样（这里 "/" 表示 end_of_doc，实际是 token [0]）：`bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/`

> [!WARNING]
> make_data.py 处理大体积 jsonl 时会非常慢，如需处理大文件请参考 [json2binidx_tool](https://github.com/Abel2076/json2binidx_tool)。

### 将 SFT messages jsonl 转换为 binidx + loss mask

SFT 数据处理使用 `data/make_sft_binidx.py`，输入是每行一个样本的 JSONL。每个样本至少需要包含 `messages`，可选包含 `tools`。`messages` 遵循常见的 chat 结构：`system`、`user`、`assistant`、`tool` 等角色按对话顺序排列；`assistant.tool_calls` 会在渲染前统一整理参数格式，字符串 JSON 会被解析成结构化参数，已有 XML 参数片段会原样保留。

示例命令：

```bash
python data/make_sft_binidx.py data/sft_part_000.jsonl data/sft_part_001.jsonl \
  --out-prefix data/sft_train \
  --vocab rwkv_vocab_v20260603.txt \
  --chat-template data/SFT/sample/chat_template.jinja \
  --pack-length 4096 \
  --num-workers 8 \
  --shuffle
```

也可以直接传入一个包含 JSONL 分片的文件夹。文件夹输入会递归展开为所有层级里的 `*.jsonl` 文件，并按路径排序：

```bash
python data/make_sft_binidx.py data/sft_shards \
  --output-prefix data/sft_train \
  --pack-length 4096 \
  --num-workers 8
```

如果你需要保持多个输入文件和每个 epoch 内的原始顺序，可以关闭打乱：

```bash
python data/make_sft_binidx.py data/sft_part_000.jsonl data/sft_part_001.jsonl \
  --out-prefix data/sft_train_ordered \
  --no-shuffle
```

使用 `--out-prefix` 或别名 `--output-prefix` 可以指定输出 binidx 的命名前缀。例如 `--out-prefix data/sft_train` 会写出 `data/sft_train.bin`、`data/sft_train.idx`、`data/sft_train.mask.bin`、`data/sft_train.mask.idx`。传入多个位置参数时必须显式指定输出前缀，因为脚本无法从多个源路径自动推导唯一名字；单个文件默认使用去掉 `.jsonl` 后缀的文件名，单个文件夹默认使用文件夹路径作为前缀。

`--num-workers` 会用在两个阶段。读取阶段以“一个 JSONL 文件”为一个任务，所以多个 JSONL 可以并发读取；单个大 JSONL 在读取阶段不会被多个 worker 拆分读取。加载完成后，模板渲染和 tokenization 会按样本并发执行。输出仍按确定的样本顺序写入，所以相同输入、`--seed`、`--shuffle` 设置会得到可复现结果。当前所有文本文件按 UTF-8 读取，JSONL 额外兼容 UTF-8 BOM，中文内容会按 UTF-8 字节映射到 token span，不会在 mask 推导中丢失。

整体流程可以抽象为：

1. 读取一个或多个 UTF-8 JSONL，过滤空行，并记录每条样本来自哪个文件和行号，便于定位坏 JSON。
2. 如果输入里有文件夹，就递归展开为所有层级的 `*.jsonl` 文件并排序。
3. 按 `--n-epoch` 重复源样本。这是写 binidx 前的离线重复：`--n-epoch 3` 表示每条源样本会被写入产物 3 次，确实会让输出数据重复 3 份。默认值是 `--n-epoch 1`，所以不想重复数据时不用传这个参数。默认每一份重复都会按 `--seed` 做确定性打乱；使用 `--no-shuffle` 时，每一份都保持输入顺序。
4. 加载权威 chat template：`data/SFT/sample/chat_template.jinja`。根目录模板不是 SFT 数据处理入口，避免误用。
5. 对每条样本先规范化工具调用参数，再用同一个 Jinja template 渲染两次：一次渲染到最后一轮 assistant 之前，用来确定条件上下文边界；一次渲染完整样本，用来得到真正写入训练集的文本。
6. 最后一轮 assistant 会被规范化为始终包含 think 标签。如果原始内容已有 think 结束标签，就保留原始 think；如果没有，就在最终回复前补一个空 think 块。历史 assistant、系统、用户、工具返回都只作为上下文。
7. loss mask 从“最后一轮 assistant 的可训练后缀”推导：assistant 内容边界之前全部为 `0`。样本里真实存在的 think 内容参与训练；无 thinking 样本自动补出的空 think 块只作为格式上下文，仍然是 `0`；可见回复、最终工具调用、assistant 结束段和真实样本结束段为 `1`。
8. 文本只 tokenize 一次。代码用 UTF-8 字节跨度记录每个 token 对应的字符区间，再把字符级可训练区间投影为 token 级 mask。这样中文、多字节符号和特殊片段都走同一套规则。
9. 如果不传 `--pack-length` 或 `--pad-length`，每条重复后的源样本会写成一个变长 binidx document，并同步写入一个同长度的 mask document，不会自动 padding。每个独立 document 末尾仍然会有真实的 `EOD_TOKEN`，并且这个 EOD 参与训练。使用 `--pack-length` 时，样本会被串接成固定长度 document，长流可以跨 document 切分，真实样本之间的分隔换行不计 loss，只有最后不足长度的尾部会 padding。使用 `--pad-length` 时不做 packing：每条源样本仍然独立成一个 document，真实 EOD 在 padding 之前，尾部补齐 token 使用 EOD token id 但 padding mask 为 `0`；如果某条样本本身超过 `--pad-length`，会直接报错。
10. 输出包含主 token 数据集和 mask sidecar：`PREFIX.bin`、`PREFIX.idx`、`PREFIX.mask.bin`、`PREFIX.mask.idx`。SFT 训练时主数据集提供 token，mask sidecar 提供哪些 token 参与 loss。

参数含义：

- `--chat-template`：SFT 渲染模板路径，默认 `data/SFT/sample/chat_template.jinja`。
- `--vocab`：tokenizer vocab，默认 `rwkv_vocab_v20260603.txt`。
- `--out-prefix` / `--output-prefix`：输出 binidx 的命名前缀，四个输出文件都由这个前缀派生。
- `--n-epoch`：离线重复数据次数，默认 `1`；大于 `1` 会让样本在产物中重复出现。
- `--seed`：打乱顺序用的随机种子；关闭 shuffle 时不影响样本顺序。
- `--shuffle` / `--no-shuffle`：是否在每个 epoch 内打乱样本，默认开启。
- `--num-workers`：并发读取、渲染和 tokenize 的 worker 数，默认 `1`。
- `--pack-length`：固定长度 packing 目标；不设置时保持一条源样本一个 document。
- `--pad-length`：不做 packing 时，把每条样本独立 padding 到固定长度；不能和 `--pack-length` 同时使用。
- `--current-date`、`--current-location`：可覆盖或注入系统消息里的日期和位置字段。

### 使用 SFT binidx 数据训练

使用 SFT 预处理产物训练时，设置 `--data_type sft_binidx`。`--data_file` 传 binidx 前缀，不带 `.bin` 或 `.idx`；训练代码默认读取 `DATA_FILE.mask` 作为 mask sidecar，也可以用 `--sft_mask_file` 显式指定另一个 mask 前缀。

在 SFT 模式下，`--epoch_steps` 和 `--epoch_count` 由用户直接控制。`epoch_steps` 表示每个 epoch 多少个 optimizer step，`epoch_count` 表示总共训练多少个 epoch。这个语义不同于预训练 `binidx`，预训练仍保留原来的 magic-prime 调度。

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
  --my_exit_tokens 0 \
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

训练端使用 next-token label，所以每个 SFT document 需要提供 `ctx_len + 1` 个 token。document 比这个短时，dataloader 会在内存里用 `--sft_pad_token_id` padding，并把 padding mask 设为 `0`；document 更长时会直接报错。为了让训练长度稳定，建议预处理时使用 `--pack-length CTX_LEN + 1` 或 `--pad-length CTX_LEN + 1`，训练时再设置 `--ctx_len CTX_LEN`。RWKV7 x070 的 `ctx_len` 需要能被 16 整除。

上面示例里的 0.4B checkpoint 是 `L24-D1024`，对应 `dim_ffn=4096`、`vocab_size=65536`、`head_size=64`，RWKV7 G1 LoRA 维度为 `64/64/32/128`。如果换用其他 checkpoint，需要先看对应架构文本，保证这些形状参数和 checkpoint 一致。

SFT mask 训练的实现框架：

1. 离线数据处理阶段只负责生成两套完全对齐的 binidx：主数据 `PREFIX.bin/.idx` 存 token id，sidecar `PREFIX.mask.bin/.idx` 存同长度的 `0/1` loss mask。
2. `train.py` 通过 `--data_type sft_binidx` 进入 SFT 分支。这个分支不使用预训练的 magic-prime 调度，而是保留用户传入的 `--epoch_steps` 和 `--epoch_count`，并把 Lightning `max_epochs` 设为 `epoch_count`，所以 SFT 会按指定 epoch 数正常结束。
3. `src/dataset.py` 的 `MyDataset` 会加载 `--data_file` 指向的 token binidx，同时默认加载 `--data_file.mask`。如果你传了 `--sft_mask_file`，就用显式 mask 前缀。初始化时会检查 token 和 mask 的 document 数量、每个 document 长度必须完全一致。
4. 每个 SFT document 最长允许 `ctx_len + 1` 个 token。短 document 会在内存中用 `--sft_pad_token_id` padding 到 `ctx_len + 1`，padding mask 始终为 `0`；长 document 直接报错，避免静默截断破坏 mask。
5. dataloader 返回三元组 `(x, y, loss_mask)`：`x = token_ids[:-1]`，`y = token_ids[1:]`，`loss_mask = raw_mask[1:]`。mask 右移是为了和 next-token label 对齐，也就是 mask 标记的是“这个 target token 是否参与 loss”。
6. `src/model.py` 的 `training_step` 根据 batch 长度分流：普通预训练 `(x, y)` 继续走原来的 fused CE 快路径；SFT `(x, y, loss_mask)` 走 `src/sft_loss.py::masked_cross_entropy`。这样 SFT 不影响预训练性能路径。
7. `masked_cross_entropy` 先用标准 CE 得到每个 token 的 loss，再只对 `loss_mask=1` 的位置求平均。如果一个 batch 的 mask 全为 `0`，返回可反传的 0 loss，避免除零和梯度图断裂。

测试流程和覆盖内容：

- 默认 CPU/单进程单测：`tests/test_sft_training.py` 覆盖 mask sidecar 加载、mask shift、padding、坏 mask、过长 document、SFT epoch 调度和 masked CE 数学正确性。
- 数据处理回归：`tests/test_sft_binidx.py` 覆盖权威 Jinja 渲染、think/no-think 规则、中文 UTF-8 span、工具调用、packing、padding、递归目录、多 JSONL 并发读取和 CLI 参数。
- CUDA smoke 第一档：`RWKV_RUN_CUDA_SFT_SMOKE=1` 会在服务器加载真实 RWKV7 checkpoint，构造 tiny SFT binidx，跑 CUDA forward/backward，验证 masked SFT loss 可以反传。
- CUDA smoke 第二档：`RWKV_RUN_TRAIN_PY_SFT_SMOKE=1` 会启动 `train.py` 跑 1 个 SFT step，覆盖 Lightning、DeepSpeed、optimizer、多卡 torchrun 链路。
- CUDA smoke 第三档：`RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1` 会保存 `rwkv-step-1.pth` 并从它恢复，覆盖 SFT 断点续训和 DeepSpeed 分片 checkpoint 加载。

当前已验证结果：

- 本地默认回归：`115 passed, 3 skipped`。
- SFT 训练相关 targeted 覆盖率：`src.dataset`、`src.sft_loss`、可单测的 `train.py` helper surface 为 `100%`。
- SFT 数据处理覆盖率：`src.sft_binidx`、`data.make_sft_binidx`、`data.tokenizer.rwkv_tokenizer` 合计 `99%`。
- 服务器 8xH800：
  - `RWKV_RUN_CUDA_SFT_SMOKE=1` -> `3 passed, 2 skipped`。
  - `RWKV_RUN_TRAIN_PY_SFT_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_2` -> `3 passed, 2 skipped`。
  - `RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_3_offload` -> `3 passed, 2 skipped`。
  - `RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_3_offload` -> `1 passed in 284.01s`。

13.3B SFT 启动脚本：

`run_13b_sft_zero3_offload.sh` 按 `model/rwkv7-g1f-13.3b.txt` 推导了模型形状：`n_layer=61`、`n_embd=4096`、`dim_ffn=16384`、`vocab_size=65536`、`head_size=64`、LoRA 维度 `192/192/128/384`。默认适配 8 卡 H800、`deepspeed_stage_3_offload` 和 SFT binidx+mask 数据。

先准备数据，注意固定长度建议使用 `ctx_len + 1`：

```bash
python data/make_sft_binidx.py /path/to/sft_jsonl_dir \
  --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
  --pack-length 8193 \
  --num-workers 32 \
  --shuffle
```

再启动 13.3B SFT：

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth \
DATA_FILE=/mnt/data/datasets/sft_train_ctx8192 \
CTX_LEN=8192 \
EPOCH_STEPS=1000 \
EPOCH_COUNT=1 \
WANDB_PROJECT=RWKV-13B-SFT \
bash run_13b_sft_zero3_offload.sh
```

如果你想验证纯 ZeRO-3 而不是 offload，可以覆盖 `STRATEGY=deepspeed_stage_3`。如果要断点续训，把 `LOAD_MODEL` 指到保存出来的 `rwkv-step-N.pth` 目录或文件，脚本仍走同一个 SFT resume 路径。

服务器上可以打开可选 CUDA smoke 测试：

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

第一条命令在进程内跑 CUDA forward/backward，验证 SFT masked loss。第二条命令启动 `train.py` 跑 1 个 SFT step，额外覆盖 Lightning、DeepSpeed 和 optimizer 链路。第三条命令会先保存 `rwkv-step-1.pth`，再从这个 step checkpoint 恢复，覆盖 SFT 断点续训；使用 DeepSpeed strategy 时，也会覆盖 DeepSpeed 分片 checkpoint 的加载。第四条命令会先产出一个 tiny SFT DeepSpeed checkpoint，然后调用合并脚本把分片 checkpoint 目录转成单文件 `.pth`，再加载并和原 ZeRO checkpoint 重构结果做等价性比较。多卡服务器可以加 `RWKV_SFT_SMOKE_DEVICES=8`，`train.py` 会自动用 torchrun 重启多卡 DeepSpeed；也可以设置 `RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3` 或 `deepspeed_stage_3_offload` 来验证不同分片模式。

如果要直接测试已有 checkpoint 目录的 pth 合并，而不是先训练 tiny checkpoint：

```bash
RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1 \
RWKV_SFT_MERGE_CHECKPOINT_DIR=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1.pth \
RWKV_SFT_MERGE_OUTPUT_FILE=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1.bf16.pth \
RWKV_SFT_MERGE_STRICT_FORWARD=1 \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_checkpoint_converts_to_pth
```

### 为指定 binidx 数据集计算 magic_prime

`data/compute_magic_prime.py` 脚本可为指定的 binidx 数据集和上下文长度（ctx_len）计算正确的 `--my_exit_tokens` 和 `--magic_prime` 值。

1. 在 `data/compute_magic_prime.py` 中修改你的训练数据集和上下文长度（`DATA_NAME` 和 `CTX_LEN`）
2. 运行脚本以获得正确的 `--my_exit_tokens` 和 `--magic_prime` 值

```
cd data/
python compute_magic_prime.py
```

最终输出类似于：

```
### Loading /home/rwkv/RWKV-LM-V7/data/demo

### /home/rwkv/RWKV-LM-V7/data/demo.bin/idx has 200499 tokens, 546 items. Dtype <class 'numpy.uint16'>

### magic_prime = 47 (for ctxlen 4096)

--my_exit_tokens 200499 --magic_prime 47 --ctx_len 4096
```

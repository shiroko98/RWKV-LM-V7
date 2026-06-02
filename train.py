########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import os
import re

logging.basicConfig(level=logging.INFO)

EPOCH_CKPT_PATTERN = re.compile(r"^rwkv-(init|\d+)\.pth$")


def parse_epoch_checkpoint_name(name: str):
    match = EPOCH_CKPT_PATTERN.match(name)
    if not match:
        return None
    token = match.group(1)
    return -1 if token == "init" else int(token)


def is_deepspeed_zero3_checkpoint_dir(path: str) -> bool:
    if not path or not os.path.isdir(path) or not path.endswith(".pth"):
        return False
    try:
        entries = set(os.listdir(path))
    except OSError:
        return False
    if "latest" in entries:
        return True
    return any(
        name.endswith("_model_states.pt") or name.endswith("_optim_states.pt")
        for name in entries
    )


def resolve_resume_checkpoint_path(path: str, strategy: str):
    if not path:
        return None
    if is_deepspeed_zero3_checkpoint_dir(path):
        if "deepspeed_stage_3" not in str(strategy):
            raise ValueError(
                f"Checkpoint directory {path} is a DeepSpeed ZeRO-3 sharded checkpoint. "
                "Please resume it with a deepspeed_stage_3* strategy."
            )
        return path
    return None

if __name__ == "__main__":
    import os
    import subprocess
    import sys
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"
    parser.add_argument("--save_every_n_steps", default=0, type=int)  # save every N real steps (0 to disable)
    parser.add_argument("--save_at_step", default=0, type=int)  # save once at this real step (0 to disable)
    parser.add_argument("--keep_last_n_checkpoints", default=0, type=int)  # keep only the most recent N numbered checkpoints (0 to disable)

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1
    parser.add_argument("--grad_clip", default=1.0, type=float) # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples

    parser.add_argument("--train_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument("--dist_timeout_sec", default=1800, type=int)
    parser.add_argument("--master_port", default=29501, type=int)

    parser.add_argument("--head_size", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_chunk", default=0, type=int) # 0 = fast, takes more VRAM; 65536 = saves 70% VRAM (when your bsz is large), slower; 4096 = saves 80% VRAM (when your bsz is large), slower
    parser.add_argument("--d_decay_lora", default=0, type=int)
    parser.add_argument("--d_aaa_lora", default=0, type=int)
    parser.add_argument("--d_mv_lora", default=0, type=int)
    parser.add_argument("--d_gate_lora", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='x070', type=str)
    parser.add_argument("--kernel", default="", type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    def _should_auto_torchrun(parsed_args) -> bool:
        if "deepspeed" not in str(parsed_args.strategy):
            return False
        if int(parsed_args.num_nodes) != 1:
            return False
        if int(parsed_args.devices) <= 1:
            return False
        if os.environ.get("LOCAL_RANK") is not None:
            return False
        if os.environ.get("WORLD_SIZE") is not None:
            return False
        return True

    if _should_auto_torchrun(args):
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.devices),
            "--master_port",
            str(args.master_port),
            os.path.abspath(__file__),
            *sys.argv[1:],
        ]
        print(f"########## Re-launching with torchrun: {' '.join(cmd)} ##########")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    ########################################################################################################

    import warnings, math, datetime, time
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["DEEPSPEED_TIMEOUT"] = str(args.dist_timeout_sec)
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_KERNEL"] = args.kernel
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = str(args.head_chunk)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    args.epoch_count = args.magic_prime // 40320
    args.epoch_steps = 40320 // args.real_bsz
    assert args.epoch_steps * args.real_bsz == 40320

    if args.train_stage >= 2:  # find latest saved model
        list_p = []
        for p in os.listdir(args.proj_dir):
            slot = parse_epoch_checkpoint_name(p)
            if slot is None:
                continue
            full_path = os.path.join(args.proj_dir, p)
            if os.path.isfile(full_path) or os.path.isdir(full_path):
                list_p += [slot]
        list_p.sort()
        max_p = list_p[-1]
        if len(list_p) > 1:
            args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
        if max_p == -1:
            args.load_model = f"{args.proj_dir}/rwkv-init.pth"
        else:
            args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            if args.warmup_steps < 0:
                args.warmup_steps = 10
        if not is_deepspeed_zero3_checkpoint_dir(args.load_model):
            args.epoch_begin = max_p + 1

    args.resume_ckpt_path = resolve_resume_checkpoint_path(args.load_model, args.strategy)
    if args.resume_ckpt_path:
        args.epoch_begin = 0

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["binidx"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0" # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    if "deepspeed" in args.strategy and dist.is_available() and not dist.is_initialized():
        local_rank = os.environ.get("LOCAL_RANK")
        world_size = os.environ.get("WORLD_SIZE")
        rank = os.environ.get("RANK")
        if local_rank is not None and world_size is not None and rank is not None:
            backend = "nccl" if args.accelerator == "gpu" else "gloo"
            if backend == "nccl":
                torch.cuda.set_device(int(local_rank))
            dist.init_process_group(
                backend=backend,
                init_method="env://",
            )

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    from src.model import RWKV
    model = RWKV(args)

    if len(args.load_model) == 0 or args.train_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    if args.resume_ckpt_path:
        rank_zero_info(f"########## Resuming trainer state from {args.resume_ckpt_path}... ##########")
        if args.load_partial == 1:
            raise ValueError("load_partial=1 is not supported when resuming from a DeepSpeed ZeRO-3 checkpoint directory.")
    else:
        rank_zero_info(f"########## Loading {args.load_model}... ##########")
        try:
            load_dict = torch.load(args.load_model, map_location="cpu", weights_only=True, mmap=True)
            load_keys = list(load_dict.keys())
            for k in load_keys:
                if k.startswith('_forward_module.'):
                    load_dict[k.replace('_forward_module.','')] = load_dict[k]
                    del load_dict[k]
        except:
            rank_zero_info(f"Bad checkpoint {args.load_model}")
            if args.train_stage >= 2:  # try again using another checkpoint
                max_p = args.my_pile_prev_p
                if max_p == -1:
                    args.load_model = f"{args.proj_dir}/rwkv-init.pth"
                else:
                    args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if not is_deepspeed_zero3_checkpoint_dir(args.load_model):
                    args.epoch_begin = max_p + 1
                args.resume_ckpt_path = resolve_resume_checkpoint_path(args.load_model, args.strategy)
                rank_zero_info(f"Trying {args.load_model}")
                if args.resume_ckpt_path:
                    load_dict = None
                else:
                    load_dict = torch.load(args.load_model, map_location="cpu", weights_only=True, mmap=True)
            else:
                raise

        if args.resume_ckpt_path:
            rank_zero_info(f"########## Resuming trainer state from {args.resume_ckpt_path}... ##########")
            if args.load_partial == 1:
                raise ValueError("load_partial=1 is not supported when resuming from a DeepSpeed ZeRO-3 checkpoint directory.")
        else:
            if args.load_partial == 1:
                load_keys = load_dict.keys()
                for k in model.state_dict():
                    if k not in load_keys:
                        load_dict[k] = model.state_dict()[k]
            model.load_state_dict(load_dict)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],
    )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    if trainer.global_rank == 0:
        print(f'### Preparing for training (loaded {args.load_model}). Please wait...')
    trainer.fit(model, data_loader, ckpt_path=args.resume_ckpt_path)

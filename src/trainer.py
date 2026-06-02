import os, math, time, datetime, subprocess, re, shutil
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

NUMBERED_CKPT_PATTERN = re.compile(r"^rwkv(?:-step)?-(\d+)\.pth$")

def is_deepspeed_strategy(strategy: str) -> bool:
    return 'deepspeed' in str(strategy)

def my_save(args, trainer, dd, ff):
    if is_deepspeed_strategy(args.strategy):
        trainer.save_checkpoint(ff, weights_only=False)
    else:
        torch.save(dd, ff)


def build_save_dict(args, pl_module):
    if args.data_type == 'wds_img':
        raw_dict = pl_module.state_dict()
        save_dict = {}
        for k in raw_dict:
            if k.startswith('encoder.') or k.startswith('decoder.'):
                save_dict[k] = raw_dict[k]
        return save_dict
    return pl_module.state_dict()


def prune_old_checkpoints(args):
    keep_last_n = getattr(args, "keep_last_n_checkpoints", 0)
    if keep_last_n <= 0:
        return

    checkpoint_entries = []
    try:
        for entry in os.scandir(args.proj_dir):
            if not (entry.is_file() or entry.is_dir()):
                continue
            if not NUMBERED_CKPT_PATTERN.match(entry.name):
                continue
            stat = entry.stat()
            checkpoint_entries.append((stat.st_mtime_ns, entry.name, entry.path, entry.is_dir()))
    except FileNotFoundError:
        return

    if len(checkpoint_entries) <= keep_last_n:
        return

    checkpoint_entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, path, is_dir in checkpoint_entries[keep_last_n:]:
        try:
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"Warning: failed to remove old checkpoint {path}: {e}")


def save_train_checkpoint(args, trainer, pl_module, file_name):
    my_save(
        args, trainer,
        build_save_dict(args, pl_module),
        file_name,
    )
    if is_deepspeed_strategy(args.strategy):
        trainer.strategy.barrier()

    if trainer.is_global_zero:
        prune_old_checkpoints(args)

    if is_deepspeed_strategy(args.strategy):
        trainer.strategy.barrier()

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._saved_step_markers = set()

    def _ensure_run_logging_state(self, trainer):
        args = self.args
        if not trainer.is_global_zero:
            return

        if not hasattr(trainer, "my_loss_sum"):
            trainer.my_loss_sum = 0
        if not hasattr(trainer, "my_loss_count"):
            trainer.my_loss_count = 0

        if not hasattr(trainer, "my_log") or getattr(trainer.my_log, "closed", False):
            trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
            run_kind = "NEW RUN" if trainer.global_step == 0 else f"RESUME RUN @ step {trainer.global_step}"
            trainer.my_log.write(f"{run_kind} {args.my_timestamp}\n{vars(args)}\n")
            try:
                print(f"\n{trainer.strategy.config}\n")
                trainer.my_log.write(f"{trainer.strategy.config}\n")
            except:
                pass
            trainer.my_log.flush()

        if len(args.wandb) > 0 and not hasattr(trainer, "my_wandb"):
            print("Login to wandb...")
            import wandb
            wandb.init(
                project=args.wandb,
                name=args.run_name + " " + args.my_timestamp,
                config=args,
                save_code=False,
            )
            trainer.my_wandb = wandb

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps

        if args.my_exit_tokens != 0: # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or is_deepspeed_strategy(args.strategy):
                    my_save(
                        args, trainer,
                        pl_module.state_dict(),
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            param_group["lr"] = lr * param_group["my_lr_scale"]

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        self._ensure_run_logging_state(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))

        if (trainer.is_global_zero) or is_deepspeed_strategy(args.strategy): # save pth
            if args.magic_prime > 0:
                if int(real_step) == int(args.magic_prime // args.real_bsz) - 1:
                    save_train_checkpoint(args, trainer, pl_module, f"{args.proj_dir}/rwkv-final.pth")

            if args.save_every_n_steps > 0 and real_step > 0 and int(real_step) % args.save_every_n_steps == 0:
                step_marker = ("every", int(real_step))
                if step_marker not in self._saved_step_markers:
                    save_train_checkpoint(args, trainer, pl_module, f"{args.proj_dir}/rwkv-step-{int(real_step)}.pth")
                    self._saved_step_markers.add(step_marker)

            if args.save_at_step > 0 and int(real_step) == int(args.save_at_step):
                step_marker = ("exact", int(real_step))
                if step_marker not in self._saved_step_markers:
                    save_train_checkpoint(args, trainer, pl_module, f"{args.proj_dir}/rwkv-step-{int(real_step)}.pth")
                    self._saved_step_markers.add(step_marker)
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        dataset.step_offset = int(trainer.global_step % args.epoch_steps)
        if trainer.is_global_zero and dataset.step_offset > 0:
            rank_zero_info(
                f"########## Resuming dataloader at step offset {dataset.step_offset}/{args.epoch_steps} "
                f"for epoch {dataset.real_epoch} ##########"
            )
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        if (trainer.is_global_zero) or is_deepspeed_strategy(args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                try:
                    save_train_checkpoint(args, trainer, pl_module, f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth")
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.train_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.train_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)

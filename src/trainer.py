import os, math, time, datetime, subprocess, re, shutil
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from src.lr_schedule import compute_sft_wsd_lr, sft_wsd_decay_enabled

NUMBERED_CKPT_PATTERN = re.compile(r"^rwkv(?:-step)?-(\d+)\.pth$")

def is_deepspeed_strategy(strategy: str) -> bool:
    return 'deepspeed' in str(strategy).lower()

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
    progress_snapshot = mark_current_batch_completed_for_checkpoint(trainer)
    try:
        my_save(
            args, trainer,
            build_save_dict(args, pl_module),
            file_name,
        )
        if is_deepspeed_strategy(args.strategy):
            trainer.strategy.barrier()

        if trainer.is_global_zero:
            prune_old_checkpoints(args)
    finally:
        restore_progress_snapshot(progress_snapshot)

    if is_deepspeed_strategy(args.strategy):
        trainer.strategy.barrier()


def _snapshot_progress_tracker(progress):
    snapshot = []
    for tracker_name in ("total", "current"):
        tracker = getattr(progress, tracker_name, None)
        if tracker is None:
            continue
        values = {}
        for attr in ("ready", "started", "processed", "completed"):
            if hasattr(tracker, attr):
                values[attr] = getattr(tracker, attr)
        snapshot.append((tracker, values))
    return snapshot


def mark_current_batch_completed_for_checkpoint(trainer):
    fit_loop = getattr(trainer, "fit_loop", None)
    epoch_loop = getattr(fit_loop, "epoch_loop", None)
    batch_progress = getattr(epoch_loop, "batch_progress", None)
    if batch_progress is None:
        return []

    snapshot = _snapshot_progress_tracker(batch_progress)
    for tracker, _ in snapshot:
        target = getattr(tracker, "processed", getattr(tracker, "ready", getattr(tracker, "completed", 0)))
        if hasattr(tracker, "completed") and tracker.completed < target:
            tracker.completed = target
    return snapshot


def restore_progress_snapshot(snapshot):
    for tracker, values in snapshot:
        for attr, value in values.items():
            setattr(tracker, attr, value)


def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(item, device) for item in batch]
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    return batch


def strategy_barrier(trainer):
    try:
        trainer.strategy.barrier()
    except Exception:
        pass


def get_global_grad_norm(trainer, pl_module):
    strategy = getattr(trainer, "strategy", None)
    strategy_model = getattr(strategy, "model", None)
    for owner in (strategy_model, getattr(pl_module, "model", None), pl_module):
        if owner is None:
            continue
        for attr in ("get_global_grad_norm", "get_grad_norm", "gradient_norm"):
            value = getattr(owner, attr, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if value is not None and not callable(value):
                try:
                    if torch.is_tensor(value):
                        return float(value.detach().float().item())
                    return float(value)
                except (TypeError, ValueError):
                    pass

    if is_deepspeed_strategy(getattr(trainer, "strategy", "")):
        return None

    parameters = getattr(pl_module, "parameters", None)
    if not callable(parameters):
        return None

    grad_norm_sq = 0.0
    has_grad = False
    for parameter in parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().float().norm(2).item()
        grad_norm_sq += grad_norm * grad_norm
        has_grad = True
    if not has_grad:
        return None
    return grad_norm_sq ** 0.5


def build_wandb_train_metrics(args, trainer, real_step, token_per_optimizer_step, t_cost, kt_s, grad_norm):
    effective_bsz = getattr(args, "effective_bsz", args.real_bsz)
    cumulative_tokens = real_step * token_per_optimizer_step
    metrics = {
        "train/loss": trainer.my_loss,
        "train/epoch_loss": trainer.my_epoch_loss,
        "train/lr": trainer.my_lr,
        "train/weight_decay": trainer.my_wd,
        "train/samples": real_step * effective_bsz,
        "train/tokens": cumulative_tokens,
        "train/tokens_b": cumulative_tokens / 1e9,
        "train/step": real_step,
    }
    if t_cost > 0:
        metrics["perf/iteration_time_sec"] = t_cost
        metrics["perf/optimizer_steps_per_sec"] = 1.0 / t_cost
        metrics["perf/tokens_per_sec"] = token_per_optimizer_step / t_cost
        metrics["perf/ktokens_per_sec"] = kt_s
        metrics["perf/samples_per_sec"] = effective_bsz / t_cost
    if grad_norm is not None:
        metrics["train/grad_norm"] = grad_norm
    return metrics


class train_callback(pl.Callback):
    def __init__(self, args, eval_loader=None):
        super().__init__()
        self.args = args
        self._saved_step_markers = set()
        self.eval_loader = eval_loader
        self._eval_step_markers = set()
        self._last_completed_real_step = None
        self._pending_loss_sum = 0.0
        self._pending_loss_count = 0

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
        if self._last_completed_real_step is None:
            self._last_completed_real_step = int(real_step)

        # LR schedule
        w_step = args.warmup_steps
        lr = compute_sft_wsd_lr(args, trainer.global_step)

        if not sft_wsd_decay_enabled(args) and args.my_exit_tokens != 0: # cosine decay
            step_bsz = getattr(args, "effective_bsz", args.real_bsz)
            real_tokens = real_step * args.ctx_len * step_bsz
            warmup_tokens = w_step * args.ctx_len * step_bsz
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

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=None):
        grad_norm = get_global_grad_norm(trainer, pl_module)
        if grad_norm is not None:
            trainer.my_grad_norm = grad_norm

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_optimizer_step = args.ctx_len * getattr(args, "effective_bsz", args.real_bsz)
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        grad_norm = getattr(trainer, "my_grad_norm", None)
        if self._last_completed_real_step is None:
            self._last_completed_real_step = max(0, int(real_step) - 1)
        step_advanced = int(real_step) > int(self._last_completed_real_step)
        t_cost = 0
        kt_s = 0

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            if not hasattr(trainer, "my_time_ns"):
                trainer.my_time_ns = t_now
            if not hasattr(trainer, "my_step_time_ns"):
                trainer.my_step_time_ns = trainer.my_time_ns
            trainer.my_time_ns = t_now
            current_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss = current_loss
            trainer.my_loss_sum += current_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self._pending_loss_sum += current_loss
            self._pending_loss_count += 1

            if step_advanced and self._pending_loss_count > 0:
                trainer.my_loss = self._pending_loss_sum / self._pending_loss_count

            if step_advanced:
                try:
                    t_cost = (t_now - trainer.my_step_time_ns) / 1e9
                    if t_cost > 0:
                        kt_s = token_per_optimizer_step / t_cost / 1000
                        self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                        self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
                except:
                    t_cost = 0
                    kt_s = 0
                trainer.my_step_time_ns = t_now
                self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
                self.log("loss", trainer.my_loss, prog_bar=True, on_step=True)

            if step_advanced and len(args.wandb) > 0:
                lll = build_wandb_train_metrics(
                    args,
                    trainer,
                    real_step,
                    token_per_optimizer_step,
                    t_cost,
                    kt_s,
                    grad_norm,
                )
                trainer.my_wandb.log(lll, step=int(real_step))

        if not step_advanced:
            return

        self._last_completed_real_step = int(real_step)
        self._pending_loss_sum = 0.0
        self._pending_loss_count = 0

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

        if self.eval_loader is not None and args.sft_eval_every_n_steps > 0 and real_step > 0:
            if int(real_step) % int(args.sft_eval_every_n_steps) == 0:
                eval_marker = int(real_step)
                if eval_marker not in self._eval_step_markers:
                    self._run_sft_eval(trainer, pl_module, eval_marker)
                    self._eval_step_markers.add(eval_marker)
                
    def _run_sft_eval(self, trainer, pl_module, real_step):
        args = self.args
        eval_loader = self.eval_loader
        if eval_loader is None:
            return

        self._ensure_run_logging_state(trainer)
        strategy_barrier(trainer)

        dataset = eval_loader.dataset
        dataset.global_rank = trainer.global_rank
        dataset.world_size = trainer.world_size
        dataset.real_epoch = 0
        dataset.step_offset = 0

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        local_stats = torch.zeros(3, dtype=torch.float64, device=device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                eval_steps = int(getattr(args, "sft_eval_steps", 0) or 0)
                if eval_steps > 0 and batch_idx >= eval_steps:
                    break
                batch = move_batch_to_device(batch, device)
                if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                    raise ValueError("SFT eval requires batches of (x, y, loss_mask).")
                if hasattr(pl_module, "sft_eval_step"):
                    loss = pl_module.sft_eval_step(batch, batch_idx)
                else:
                    loss = pl_module.training_step(batch, batch_idx)
                mask_tokens = batch[2].float().sum()
                local_stats[0] += loss.detach().float().to(dtype=torch.float64) * mask_tokens.to(dtype=torch.float64)
                local_stats[1] += mask_tokens.to(dtype=torch.float64)
                local_stats[2] += batch[0].shape[0]

        if was_training:
            pl_module.train()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)
        strategy_barrier(trainer)

        if local_stats[1].item() <= 0:
            val_loss = 0.0
            val_ppl = 1.0
        else:
            val_loss = (local_stats[0] / local_stats[1]).item()
            val_ppl = math.exp(min(val_loss, 20.0))

        if trainer.is_global_zero:
            eval_docs = int(local_stats[2].item())
            eval_tokens = int(local_stats[1].item())
            msg = (
                f"eval step {int(real_step)} loss {val_loss:.6f} ppl {val_ppl:.4f} "
                f"mask_tokens {eval_tokens} docs {eval_docs}"
            )
            rank_zero_info(f"########## SFT {msg} ##########")
            if hasattr(trainer, "my_log") and not getattr(trainer.my_log, "closed", False):
                trainer.my_log.write(msg + f" {datetime.datetime.now()}\n")
                trainer.my_log.flush()
            if len(args.wandb) > 0 and hasattr(trainer, "my_wandb"):
                trainer.my_wandb.log(
                    {
                        "eval/loss": val_loss,
                        "eval/ppl": val_ppl,
                        "eval/mask_tokens": eval_tokens,
                        "eval/docs": eval_docs,
                    },
                    step=int(real_step),
                )

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

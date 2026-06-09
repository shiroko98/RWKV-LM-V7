import math


LR_WSD_DECAY_STYLES = {"none", "linear", "cosine"}


def sft_wsd_decay_enabled(args) -> bool:
    style = str(getattr(args, "lr_wsd_decay_style", "cosine") or "cosine").lower()
    iters = int(getattr(args, "lr_wsd_decay_iters", 0) or 0)
    return getattr(args, "data_type", "") == "sft_binidx" and iters > 0 and style != "none"


def compute_sft_wsd_lr(args, optimizer_step: int) -> float:
    style = str(getattr(args, "lr_wsd_decay_style", "cosine") or "cosine").lower()
    if style not in LR_WSD_DECAY_STYLES:
        raise ValueError(f"lr_wsd_decay_style must be one of {sorted(LR_WSD_DECAY_STYLES)}.")

    decay_iters = int(getattr(args, "lr_wsd_decay_iters", 0) or 0)
    if decay_iters < 0:
        raise ValueError("lr_wsd_decay_iters must be non-negative.")
    if not sft_wsd_decay_enabled(args):
        return float(args.lr_init)

    total_steps = int(args.epoch_count * args.epoch_steps)
    if total_steps <= 0:
        raise ValueError("SFT WSD LR schedule requires positive epoch_steps and epoch_count.")

    schedule_step = optimizer_step - int(getattr(args, "epoch_begin", 0) * args.epoch_steps)
    schedule_step = max(0, schedule_step)
    decay_steps = min(decay_iters, total_steps)
    decay_start = total_steps - decay_steps
    if schedule_step < decay_start:
        return float(args.lr_init)
    if decay_steps <= 1 or schedule_step >= total_steps - 1:
        return float(args.lr_final)

    progress = (schedule_step - decay_start) / (decay_steps - 1)
    progress = max(0.0, min(1.0, progress))
    if style == "linear":
        return float(args.lr_init + (args.lr_final - args.lr_init) * progress)
    return float(args.lr_final + 0.5 * (args.lr_init - args.lr_final) * (1.0 + math.cos(math.pi * progress)))

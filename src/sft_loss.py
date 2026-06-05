import torch
from torch.nn import functional as F


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocab].")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape [batch, time].")
    if loss_mask.shape != targets.shape:
        raise ValueError("loss_mask must have the same shape as targets.")

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    mask = loss_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
    mask_sum = mask.sum()
    if mask_sum <= 0:
        return per_token_loss.sum() * 0
    return (per_token_loss * mask).sum() / mask_sum

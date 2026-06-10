import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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


def _masked_head_loss_chunk(
    head,
    flat_hidden: torch.Tensor,
    flat_targets: torch.Tensor,
    flat_mask: torch.Tensor,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    chunk_hidden = flat_hidden.index_select(0, active_indices)
    chunk_targets = flat_targets.index_select(0, active_indices)
    chunk_weights = flat_mask.index_select(0, active_indices).to(dtype=torch.float32)
    if callable(head):
        logits = head(chunk_hidden).float()
    else:
        logits = F.linear(chunk_hidden, head).float()
    per_token_loss = F.cross_entropy(logits, chunk_targets, reduction="none")
    return (per_token_loss * chunk_weights).sum()


def masked_head_cross_entropy(
    hidden: torch.Tensor,
    head,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    chunk_size: int = 1024,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, time, hidden_size].")
    weight = getattr(head, "weight", head)
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("head weight must have shape [vocab, hidden_size].")
    if targets.shape != hidden.shape[:2]:
        raise ValueError("targets must have shape [batch, time].")
    if loss_mask.shape != targets.shape:
        raise ValueError("loss_mask must have the same shape as targets.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(device=hidden.device, dtype=torch.float32)
    active_indices = torch.nonzero(flat_mask != 0, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        if callable(head):
            return (hidden.sum() + head(flat_hidden[:1]).sum()) * 0
        return (hidden.sum() + weight.sum()) * 0

    active_weight_sum = flat_mask.index_select(0, active_indices).sum()
    loss_sum = None
    use_checkpoint = torch.is_grad_enabled() and checkpoint_chunks and (flat_hidden.requires_grad or weight.requires_grad)
    for start in range(0, active_indices.numel(), chunk_size):
        chunk_indices = active_indices[start : start + chunk_size]
        if use_checkpoint:
            def loss_chunk(
                checkpoint_hidden: torch.Tensor,
                checkpoint_targets: torch.Tensor,
                checkpoint_mask: torch.Tensor,
                checkpoint_indices: torch.Tensor,
            ) -> torch.Tensor:
                return _masked_head_loss_chunk(
                    head,
                    checkpoint_hidden,
                    checkpoint_targets,
                    checkpoint_mask,
                    checkpoint_indices,
                )

            chunk_loss = checkpoint(
                loss_chunk,
                flat_hidden,
                flat_targets,
                flat_mask,
                chunk_indices,
                use_reentrant=False,
            )
        else:
            chunk_loss = _masked_head_loss_chunk(head, flat_hidden, flat_targets, flat_mask, chunk_indices)
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss

    return loss_sum / active_weight_sum.to(dtype=loss_sum.dtype)

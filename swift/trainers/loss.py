from typing import Callable, Optional

import torch
from torch.nn import CrossEntropyLoss

class LossName:
    long_ce = 'long-ce'
    loss_scale = 'loss-scale'
    per_sample_loss = 'per-sample-loss'


LOSS_MAPPING = {}


def register_loss_func(loss_name: str, loss_func: Optional[Callable] = None):
    loss_info = {}

    if loss_func is not None:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_name] = loss_info
        return

    def _register_loss_func(loss_func: Callable) -> Callable:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_name] = loss_info
        return loss_func

    return _register_loss_func


def ce_loss_func(outputs, labels):
    logits = outputs.logits
    device = logits.device
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    # Save memory
    masks = shift_labels != -100
    shift_logits = shift_logits[masks]
    shift_labels = shift_labels[masks]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)
    return loss, masks


class LongCrossEntropy:
    """Assign higher weight to long text."""

    def __init__(self, length_smooth: float = 0.9):
        self._s_length = 0
        self._norm_factor = 0
        self._smoothing = length_smooth

    def __call__(self, outputs, labels, num_items_in_batch=None) -> torch.Tensor:
        # moving average
        loss, masks = ce_loss_func(outputs, labels)
        if num_items_in_batch is not None:
            # The gradient accumulation equivalent to mini_batch for transformers >= 4.46 and fallback behavior.
            return loss.sum() / num_items_in_batch
        self._s_length = self._s_length * self._smoothing + loss.shape[0]
        self._norm_factor = self._norm_factor * self._smoothing + 1
        loss = loss.sum() / (self._s_length / self._norm_factor)
        return loss


register_loss_func(LossName.long_ce, LongCrossEntropy())


@register_loss_func(LossName.loss_scale)
def loss_scale_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    loss, masks = ce_loss_func(outputs, labels)
    if loss_scale is not None:
        shift_scale = loss_scale[..., 1:].to(masks.device)
        shift_scale = shift_scale[masks]
        loss = (shift_scale * loss)
    if num_items_in_batch is None:
        loss = loss.mean()
    else:
        # compat transformers>=4.46
        loss = loss.sum() / num_items_in_batch
    return loss


@register_loss_func(LossName.per_sample_loss)
def per_sample_loss_func(input_ids, outputs, labels) -> torch.Tensor:
    new_labels = []
    boi_token_id = 151339
    eoi_token_id = 151340

    for i in range(len(input_ids)):
        input_id = input_ids[i].tolist()
        boi_token_pos, eoi_token_pos = input_id.index(boi_token_id), input_id.index(
            eoi_token_id)
        assert eoi_token_pos - boi_token_pos == 2

        new_labels.append(torch.cat(
            (
                labels[i, :boi_token_pos + 1],
                torch.tensor([-100]).to(labels.device).to(labels.dtype).repeat(1600),
                labels[i, eoi_token_pos:])))
    
    labels = torch.stack(new_labels, dim=0)
    lm_logits = outputs['logits']
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    with torch.no_grad():
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        non_ignored_mask = (shift_labels != -100).type_as(loss)
        loss = loss * non_ignored_mask
        non_ignored_count = non_ignored_mask.sum(dim=1).clamp(min=1)
        loss_per_sample = loss.sum(dim=1) / non_ignored_count
  
    return loss_per_sample


def get_loss_func(loss_name: Optional[str]) -> Optional[Callable]:
    if loss_name is None:
        return None
    return LOSS_MAPPING[loss_name]['loss_func']

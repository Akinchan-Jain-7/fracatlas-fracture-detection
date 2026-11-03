"""Learning-rate schedule: linear warmup, then cosine decay.

- Warmup: the new single-logit head starts random, so its first gradients are
  large and noisy. Ramping the LR up over the first epoch keeps them from
  wrecking the pretrained backbone weights.
- Cosine decay: big steps early to make progress, small steps late to settle
  into a good minimum, without hand-picking decay epochs.

Stepped once per *batch*, so the curve is smooth regardless of epoch count.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(
    optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01
) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)

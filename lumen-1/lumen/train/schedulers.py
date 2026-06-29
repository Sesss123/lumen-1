"""Learning rate schedules."""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_wsd_schedule(
    optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    peak_lr: float = 3e-4,
    min_lr: float = 3e-5,
):
    """Warmup-Stable-Decay schedule for pretraining."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(num_warmup_steps, 1)
        if step < num_warmup_steps + num_stable_steps:
            return 1.0
        decay_step = step - num_warmup_steps - num_stable_steps
        if decay_step >= num_decay_steps:
            return min_lr / peak_lr
        progress = decay_step / max(num_decay_steps, 1)
        return (min_lr / peak_lr) + 0.5 * (1 - min_lr / peak_lr) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1):
    """Cosine decay for SFT/alignment."""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(num_warmup_steps, 1)
        progress = (step - num_warmup_steps) / max(num_training_steps - num_warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

"""Code reproducibility."""

import random
import numpy as np
import torch


def fix_seeds(seed_value: int = 42, device: str = "cpu") -> None:
    """Source 'https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'.

    :param seed_value: random state value
    :param device: device for which seeds would be fixed
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != "cpu":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from __future__ import annotations

from typing import Any
from .rca_reward import rca_reward

def score_rca_exact(full_state: dict[str, Any], gt_labels: list[Any], pred_labels: list[Any], **kwargs: Any) -> dict[str, Any]:
    return rca_reward(full_state, gt_labels, pred_labels, **kwargs)

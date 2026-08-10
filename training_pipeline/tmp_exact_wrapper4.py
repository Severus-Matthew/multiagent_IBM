from typing import Any

def f(full_state: dict[str, Any], gt_labels: list[Any], pred_labels: list[Any], **kwargs: Any) -> dict[str, Any]:
    out = {}
    comps = dict(out.get('components') or {})
    verdict = bool(comps.get('exact_set_match'))
    out['success'] = verdict
    return out

import torch

CLIP_NORM = 65.0
TRIM_FRAC = 0.2
SIGN_THRESHOLD = 0.6


def _float_keys(state_dicts):
    return [k for k in state_dicts[0] if state_dicts[0][k].dtype.is_floating_point]


def _copy_non_float(state_dicts, result):
    for key in state_dicts[0]:
        if key not in result:
            result[key] = state_dicts[0][key].clone()
    return result


def _to_updates(state_dicts, global_state):
    return [{k: sd[k].float() - global_state[k].float() for k in _float_keys(state_dicts)}
            for sd in state_dicts]


def _flat_norm(update):
    return torch.sqrt(sum((v ** 2).sum() for v in update.values()))


def fedavg(state_dicts, weights, **kwargs):
    total = sum(weights)
    fractions = [w / total for w in weights]

    result = {}
    for key in _float_keys(state_dicts):
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        frac = torch.tensor(fractions, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
        result[key] = (stacked * frac).sum(dim=0).to(state_dicts[0][key].dtype)

    return _copy_non_float(state_dicts, result)


def norm_clipping(state_dicts, weights, global_state=None, clip_norm=CLIP_NORM, **kwargs):
    """
    Scale every client update down to at most clip_norm before averaging.
    Applied identically to all clients, so no client is singled out.
    """
    updates = _to_updates(state_dicts, global_state)

    for update in updates:
        norm = _flat_norm(update)
        scale = min(1.0, clip_norm / (norm.item() + 1e-9))
        for key in update:
            update[key] *= scale

    total = sum(weights)
    result = {}
    for key in updates[0]:
        stacked = torch.stack([u[key] for u in updates])
        frac = torch.tensor([w / total for w in weights],
                            device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
        avg_update = (stacked * frac).sum(dim=0)
        result[key] = (global_state[key].float() + avg_update).to(state_dicts[0][key].dtype)

    return _copy_non_float(state_dicts, result)


def trimmed_mean(state_dicts, weights, trim_frac=TRIM_FRAC, **kwargs):
    """
    Per coordinate, drop the highest and lowest values, then average the rest.
    """
    n = len(state_dicts)
    k = max(1, int(n * trim_frac))

    result = {}
    for key in _float_keys(state_dicts):
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        kept = sorted_vals[k:n - k] if n - 2 * k > 0 else sorted_vals
        result[key] = kept.mean(dim=0).to(state_dicts[0][key].dtype)

    return _copy_non_float(state_dicts, result)


def invariant_aggregator(state_dicts, weights, global_state=None,
                         sign_threshold=SIGN_THRESHOLD, trim_frac=TRIM_FRAC, **kwargs):

    updates = _to_updates(state_dicts, global_state)
    n = len(updates)
    k = max(1, int(n * trim_frac))

    result = {}
    for key in updates[0]:
        stacked = torch.stack([u[key] for u in updates])

        signs = torch.sign(stacked)
        agreement = signs.sum(dim=0).abs() / n
        mask = (agreement >= sign_threshold).float()

        sorted_vals, _ = torch.sort(stacked, dim=0)
        kept = sorted_vals[k:n - k] if n - 2 * k > 0 else sorted_vals
        avg_update = kept.mean(dim=0) * mask

        result[key] = (global_state[key].float() + avg_update).to(state_dicts[0][key].dtype)

    return _copy_non_float(state_dicts, result)


METHODS = {
    "fedavg": fedavg,
    "clipping": norm_clipping,
    "trimmed": trimmed_mean,
    "invariant": invariant_aggregator,
}


def aggregate(state_dicts, weights, method="fedavg", **kwargs):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}. options: {list(METHODS)}")
    return METHODS[method](state_dicts, weights, **kwargs)
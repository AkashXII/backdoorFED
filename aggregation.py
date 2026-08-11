import torch


def fedavg(state_dicts, weights):
    total = sum(weights)
    fractions = [w / total for w in weights]

    result = {}
    for key in state_dicts[0]:
        if not state_dicts[0][key].dtype.is_floating_point:
            result[key] = state_dicts[0][key].clone()
            continue

        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        frac = torch.tensor(fractions, device=stacked.device).view(-1, *([1] * (stacked.dim() - 1)))
        result[key] = (stacked * frac).sum(dim=0).to(state_dicts[0][key].dtype)

    return result


def aggregate(state_dicts, weights, method="fedavg", **kwargs):
    if method == "fedavg":
        return fedavg(state_dicts, weights)
    raise ValueError(f"unknown method: {method}")
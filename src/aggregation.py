import torch

CLIP_NORM = 65.0
TRIM_FRAC = 0.2
SIGN_THRESHOLD = 0.5


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
    """
    Keep only coordinates where clients mostly agree on the direction of change,
    then combine those with a trimmed mean. Coordinates without agreement are
    left at their current value.
    """
    updates = _to_updates(state_dicts, global_state)
    n = len(updates)
    k = max(1, int(n * trim_frac))

    result = {}
    for key in updates[0]:
        stacked = torch.stack([u[key] for u in updates])

        signs = torch.sign(stacked)
        agreement = signs.sum(dim=0).abs() / n
        mask = (agreement >= sign_threshold - 1e-6).float()

        sorted_vals, _ = torch.sort(stacked, dim=0)
        kept = sorted_vals[k:n - k] if n - 2 * k > 0 else sorted_vals
        avg_update = kept.mean(dim=0) * mask

        result[key] = (global_state[key].float() + avg_update).to(state_dicts[0][key].dtype)

    return _copy_non_float(state_dicts, result)


def flame(state_dicts, weights, global_state=None, client_names=None,
          noise_std=0.001, verbose=False, **kwargs):
    """
    FLAME (USENIX Security 2022): cluster updates by cosine direction with
    HDBSCAN, keep the largest cluster (the benign majority), clip survivors to
    the median norm, average, and add a little noise.

    Falls back gracefully when there are too few points to cluster.
    """
    import numpy as np

    updates = _to_updates(state_dicts, global_state)
    n = len(updates)

    flat = torch.stack([
        torch.cat([u[k].flatten() for k in updates[0]]) for u in updates
    ]).cpu().numpy()

    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / (norms + 1e-9)

    try:
        import hdbscan
        min_cluster = max(2, n // 2 + 1)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric="euclidean",
                                    allow_single_cluster=True)
        labels = clusterer.fit_predict(normed)
    except Exception:
        labels = np.zeros(n, dtype=int)

    if (labels >= 0).sum() == 0:
        keep_idx = list(range(n))
    else:
        valid = labels[labels >= 0]
        majority = np.bincount(valid).argmax()
        keep_idx = [i for i in range(n) if labels[i] == majority]

    if verbose:
        names = client_names if client_names else list(range(n))
        rejected = [names[i] for i in range(n) if i not in keep_idx]
        print(f"      FLAME: {len(keep_idx)}/{n} kept, rejected {rejected}, "
              f"clusters {sorted(set(labels.tolist()))}")

    flat_norms = np.linalg.norm(flat, axis=1)
    clip_val = float(np.median(flat_norms))

    result = {}
    for key in updates[0]:
        acc = None
        for i in keep_idx:
            scale = min(1.0, clip_val / (flat_norms[i] + 1e-9))
            contrib = updates[i][key] * scale
            acc = contrib if acc is None else acc + contrib
        avg = acc / len(keep_idx)
        noise = torch.randn_like(avg) * noise_std * clip_val
        result[key] = (global_state[key].float() + avg + noise).to(state_dicts[0][key].dtype)

    return _copy_non_float(state_dicts, result)


def multi_krum(state_dicts, weights, global_state=None, n_attackers=1,
               n_select=None, **kwargs):
    """
    Multi-Krum (Blanchard et al. 2017). For each update, sum the squared
    distances to its n - f - 2 nearest neighbours; that sum is its score.
    Keep the update(s) with the lowest scores and average them.

    Low score = sits in a tight neighbourhood of other updates = presumed benign.
    """
    import numpy as np

    updates = _to_updates(state_dicts, global_state)
    n = len(updates)

    flat = torch.stack([
        torch.cat([u[k].flatten() for k in updates[0]]) for u in updates
    ])

    dists = torch.cdist(flat, flat) ** 2
    dists = dists.cpu().numpy()
    np.fill_diagonal(dists, np.inf)

    n_neighbours = max(1, n - n_attackers - 2)
    scores = []
    for i in range(n):
        nearest = np.sort(dists[i])[:n_neighbours]
        scores.append(nearest.sum())

    if n_select is None:
        n_select = max(1, n - n_attackers)
    keep_idx = list(np.argsort(scores)[:n_select])

    total = sum(weights[i] for i in keep_idx)
    result = {}
    for key in updates[0]:
        acc = None
        for i in keep_idx:
            contrib = updates[i][key] * (weights[i] / total)
            acc = contrib if acc is None else acc + contrib
        result[key] = (global_state[key].float() + acc).to(state_dicts[0][key].dtype)

    return _copy_non_float(state_dicts, result)


CLIENT_SOURCE = {
    "rsna_a": "rsna", "rsna_b": "rsna", "rsna_c": "rsna",
    "kerm_a": "kermany", "kerm_b": "kermany",
}


def source_aware(state_dicts, weights, global_state=None, client_names=None,
                 inner_method="trimmed", **kwargs):
    """
    Group clients by data source, aggregate robustly within each group, then
    average the group results.

    The point: a Kermany client only looks anomalous when compared against RSNA
    clients. Compared against other Kermany clients it looks normal, so robust
    aggregation inside a group is not confounded by domain shift.
    """
    if client_names is None:
        raise ValueError("source_aware needs client_names")

    groups = {}
    for i, name in enumerate(client_names):
        groups.setdefault(CLIENT_SOURCE[name], []).append(i)

    group_states, group_weights = [], []
    for source, idxs in groups.items():
        sub_states = [state_dicts[i] for i in idxs]
        sub_weights = [weights[i] for i in idxs]

        if len(sub_states) == 1:
            group_states.append(sub_states[0])
        else:
            group_states.append(
                METHODS[inner_method](sub_states, sub_weights,
                                      global_state=global_state)
            )
        group_weights.append(sum(sub_weights))

    return fedavg(group_states, group_weights)


METHODS = {
    "fedavg": fedavg,
    "clipping": norm_clipping,
    "trimmed": trimmed_mean,
    "invariant": invariant_aggregator,
    "source_aware": source_aware,
    "flame": flame,
    "multikrum": multi_krum,
}


def aggregate(state_dicts, weights, method="fedavg", **kwargs):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}. options: {list(METHODS)}")
    return METHODS[method](state_dicts, weights, **kwargs)
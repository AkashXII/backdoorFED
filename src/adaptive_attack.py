
import argparse, copy, time
import numpy as np, pandas as pd, torch

from aggregation import aggregate, _to_updates, _float_keys
from fed_train import (make_client_loaders, make_test_loaders, train_local, evaluate,
                       select_clients, build_model, ROUNDS, ATTACK_START_ROUND,
                       COUNTER_FRAC, ATTACKER, POISON_FRAC)


def adapt_update(attacker_sd, global_state, prev_global_update, align_frac=0.9):
    """
    Modify the attacker's state dict so that on a fraction of coordinates its
    update sign matches the previous global update sign. Where signs already
    agree, keep the backdoor magnitude; where they disagree, damp it.
    """
    if prev_global_update is None:
        return attacker_sd

    new_sd = {}
    for key in attacker_sd:
        if key not in _float_keys([attacker_sd]):
            new_sd[key] = attacker_sd[key].clone()
            continue

        upd = attacker_sd[key].float() - global_state[key].float()
        ref = prev_global_update[key]

        agree = (torch.sign(upd) == torch.sign(ref))
        # where signs disagree, flip the attacker's update to agree (damped)
        damp = torch.where(agree, torch.ones_like(upd), torch.full_like(upd, -align_frac))
        new_upd = upd * damp

        new_sd[key] = (global_state[key].float() + new_upd).to(attacker_sd[key].dtype)
    return new_sd


def main(method, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    clean = make_client_loaders(seed, attack=False, poison_frac=POISON_FRAC)
    poison = make_client_loaders(seed, attack=True, poison_frac=POISON_FRAC,
                                 counter_frac=COUNTER_FRAC)
    test_loaders = make_test_loaders()
    names = sorted(clean)

    model = build_model()
    gs = copy.deepcopy(model.state_dict())
    prev_global_update = None

    for rnd in range(1, ROUNDS + 1):
        sel = select_clients(rng, names, True, rnd)
        active = rnd >= ATTACK_START_ROUND and ATTACKER in sel

        prev_gs = copy.deepcopy(gs)
        sds, ws = [], []
        for name in sel:
            model.load_state_dict(gs)
            loader = poison[name] if (active and name == ATTACKER) else clean[name]
            train_local(model, loader, 2)
            sd = copy.deepcopy(model.state_dict())
            if active and name == ATTACKER:
                sd = adapt_update(sd, gs, prev_global_update)
            sds.append(sd); ws.append(len(loader.dataset))

        gs = aggregate(sds, ws, method=method, global_state=gs, client_names=list(sel))

        prev_global_update = {k: gs[k].float() - prev_gs[k].float() for k in _float_keys([gs])}

    model.load_state_dict(gs)
    s = evaluate(model, test_loaders)
    print(f"{method}  seed={seed}  AUROC {s['all']:.3f}  ASR {s['asr']:.3f}  (ADAPTIVE)")
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="invariant")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.method, a.seed)
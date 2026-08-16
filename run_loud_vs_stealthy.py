import argparse
import copy
import itertools
import time

import numpy as np
import pandas as pd
import torch

from aggregation import aggregate
from fed_train import (
    make_client_loaders, make_test_loaders, train_local, evaluate, select_clients,
    build_model, ROUNDS, ATTACK_START_ROUND, COUNTER_FRAC, ATTACKER,
)

METHODS = ["fedavg", "clipping", "trimmed", "invariant"]
POISON_RATES = {"stealthy": 0.05, "loud": 0.75}
SEEDS = [0, 1, 2]
OUT_CSV = "loud_vs_stealthy.csv"


def run_one(method, poison_frac, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    clean = make_client_loaders(seed, attack=False, poison_frac=poison_frac)
    poison = make_client_loaders(seed, attack=True, poison_frac=poison_frac,
                                 counter_frac=COUNTER_FRAC)
    test_loaders = make_test_loaders()
    client_names = sorted(clean)

    model = build_model()
    global_state = copy.deepcopy(model.state_dict())

    for rnd in range(1, ROUNDS + 1):
        selected = select_clients(rng, client_names, True, rnd)
        active = (rnd >= ATTACK_START_ROUND) and (ATTACKER in selected)

        state_dicts, weights = [], []
        for name in selected:
            model.load_state_dict(global_state)
            loader = poison[name] if (active and name == ATTACKER) else clean[name]
            train_local(model, loader, LOCAL_EPOCHS := 2)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            weights.append(len(loader.dataset))

        global_state = aggregate(state_dicts, weights, method=method,
                                 global_state=global_state,
                                 client_names=list(selected))

    model.load_state_dict(global_state)
    return evaluate(model, test_loaders)


def main(seeds):
    rows = []
    combos = list(itertools.product(POISON_RATES.items(), METHODS, seeds))

    for i, ((label, pf), method, seed) in enumerate(combos, 1):
        start = time.time()
        scores = run_one(method, pf, seed)
        rows.append({"attack": label, "poison_frac": pf, "method": method,
                     "seed": seed, "auroc": scores["all"], "asr": scores["asr"]})
        print(f"[{i}/{len(combos)}] {label:8s} {method:10s} seed={seed}  "
              f"AUROC {scores['all']:.3f}  ASR {scores['asr']:.3f}  ({time.time()-start:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    print("\n=== mean ASR over seeds ===")
    pivot = df.groupby(["attack", "method"])["asr"].mean().unstack()
    print(pivot.round(3).to_string())
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    main(parser.parse_args().seeds)

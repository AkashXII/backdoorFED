import argparse
import copy

import numpy as np
import pandas as pd
import torch

from aggregation import aggregate
from fed_train import (
    make_client_loaders, make_test_loaders, train_local, evaluate, select_clients,
    build_model, ROUNDS, CLIENTS_PER_ROUND, LOCAL_EPOCHS,
    ATTACKER, ATTACK_START_ROUND, COUNTER_FRAC,
)

POISON_FRACS =  [0.05, 0.10, 0.20, 0.40, 0.60]


def run_one(seed, poison_frac):
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
            train_local(model, loader, LOCAL_EPOCHS)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            weights.append(len(loader.dataset))

        global_state = aggregate(state_dicts, weights, method="fedavg")

    model.load_state_dict(global_state)
    scores = evaluate(model, test_loaders)
    return scores


def main(seed):
    rows = []
    for pf in POISON_FRACS:
        scores = run_one(seed, pf)
        rows.append({"poison_frac": pf, "auroc": scores["all"], "asr": scores["asr"]})
        print(f"poison_frac {pf:.2f}  AUROC {scores['all']:.3f}  ASR {scores['asr']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"sweep_poison_seed{seed}.csv", index=False)
    print(f"\nsaved sweep_poison_seed{seed}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
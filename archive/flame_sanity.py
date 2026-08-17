import argparse
import copy

import numpy as np
import torch

from aggregation import flame
from fed_train import (
    make_client_loaders, train_local, select_clients, build_model,
    ROUNDS, ATTACK_START_ROUND, COUNTER_FRAC, ATTACKER, POISON_FRAC,
)

CHECK_FROM = 15


def main(seed, poison_frac):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    clean = make_client_loaders(seed, attack=False, poison_frac=poison_frac)
    poison = make_client_loaders(seed, attack=True, poison_frac=poison_frac,
                                 counter_frac=COUNTER_FRAC)
    client_names = sorted(clean)

    model = build_model()
    global_state = copy.deepcopy(model.state_dict())

    print(f"seed {seed}  poison_frac {poison_frac}  attacker {ATTACKER}\n")
    attacker_caught, attacker_rounds = 0, 0

    for rnd in range(1, ROUNDS + 1):
        selected = select_clients(rng, client_names, True, rnd)
        active = (rnd >= ATTACK_START_ROUND) and (ATTACKER in selected)

        state_dicts, weights = [], []
        for name in selected:
            model.load_state_dict(global_state)
            loader = poison[name] if (active and name == ATTACKER) else clean[name]
            train_local(model, loader, 2)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            weights.append(len(loader.dataset))

        verbose = rnd >= CHECK_FROM
        if verbose:
            print(f"round {rnd}  attacker_present={active}")

        global_state = flame(state_dicts, weights, global_state=global_state,
                             client_names=list(selected), verbose=verbose)

    print("\nsanity check complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poison_frac", type=float, default=0.10)
    args = parser.parse_args()
    main(args.seed, args.poison_frac)

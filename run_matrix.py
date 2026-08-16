import argparse
import copy
import itertools
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from aggregation import aggregate, _to_updates, _flat_norm
from data import (ChestXrayDataset, load_manifest, split_by_patient,
                  split_client_train_test, eval_transform)
from fed_train import (
    make_client_loaders, make_test_loaders, train_local, evaluate, select_clients,
    build_model, ROUNDS, CLIENTS_PER_ROUND, LOCAL_EPOCHS,
    ATTACKER, ATTACK_START_ROUND, COUNTER_FRAC, SPLIT_SEED,
)
from train_central import predict

METHODS = ["trimmed", "invariant", "multikrum", "flame"]
ATTACK_SETTINGS = [("none", 0.0), ("stealthy", 0.10), ("loud", 0.40)]
SEEDS = [0, 1, 2, 3, 4]

OUT_CSV = "matrix_results_part2.csv"
NORMS_CSV = "matrix_norms_part2.csv"

def per_client_test_loaders(seed):
    clients_df = pd.read_csv(f"clients_seed{seed}.csv")
    _, client_test = split_client_train_test(clients_df, test_frac=0.2, seed=seed)

    loaders = {}
    for name, group in client_test.groupby("client"):
        loaders[name] = DataLoader(ChestXrayDataset(group, eval_transform),
                                   batch_size=64, num_workers=2)
    return loaders


def run_one(method, attack_label, poison_frac, seed, norm_log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    attack = poison_frac > 0
    clean = make_client_loaders(seed, attack=False, poison_frac=poison_frac)
    poison = make_client_loaders(seed, attack=True, poison_frac=poison_frac,
                                 counter_frac=COUNTER_FRAC) if attack else None
    test_loaders = make_test_loaders()
    client_test = per_client_test_loaders(seed)
    client_names = sorted(clean)

    model = build_model()
    global_state = copy.deepcopy(model.state_dict())

    for rnd in range(1, ROUNDS + 1):
        selected = select_clients(rng, client_names, attack, rnd)
        active = attack and (rnd >= ATTACK_START_ROUND) and (ATTACKER in selected)

        state_dicts, weights = [], []
        for name in selected:
            model.load_state_dict(global_state)
            loader = poison[name] if (active and name == ATTACKER) else clean[name]
            train_local(model, loader, LOCAL_EPOCHS)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            weights.append(len(loader.dataset))

        updates = _to_updates(state_dicts, global_state)
        for name, upd in zip(selected, updates):
            norm_log.append({
                "method": method, "attack": attack_label, "seed": seed, "round": rnd,
                "client": name, "is_attacker": bool(active and name == ATTACKER),
                "norm": _flat_norm(upd).item(),
            })

        global_state = aggregate(state_dicts, weights, method=method,
                                 global_state=global_state,
                                 client_names=list(selected))

    model.load_state_dict(global_state)
    scores = evaluate(model, test_loaders)

    for name, loader in client_test.items():
        probs, labels = predict(model, loader)
        try:
            scores[f"auroc_{name}"] = roc_auc_score(labels, probs)
        except ValueError:
            scores[f"auroc_{name}"] = float("nan")

    return scores


def main(seeds):
    all_rows, norm_log = [], []
    combos = list(itertools.product(METHODS, ATTACK_SETTINGS, seeds))

    for i, (method, (attack_label, pf), seed) in enumerate(combos, 1):
        start = time.time()
        scores = run_one(method, attack_label, pf, seed, norm_log)

        row = {"method": method, "attack": attack_label, "poison_frac": pf, "seed": seed}
        row.update(scores)
        all_rows.append(row)

        print(f"[{i}/{len(combos)}] {method:10s} {attack_label:8s} seed={seed}  "
              f"AUROC {scores['all']:.3f}  ASR {scores['asr']:.3f}  "
              f"({time.time()-start:.0f}s)")

        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        pd.DataFrame(norm_log).to_csv(NORMS_CSV, index=False)

    print(f"\nsaved {OUT_CSV} and {NORMS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    main(parser.parse_args().seeds)
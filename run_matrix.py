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
from data import ChestXrayDataset, load_manifest, split_by_patient, eval_transform
from fed_train import (
    make_client_loaders, make_test_loaders, train_local, evaluate, select_clients,
    build_model, ROUNDS, CLIENTS_PER_ROUND, LOCAL_EPOCHS,
    ATTACKER, ATTACK_START_ROUND, COUNTER_FRAC, POISON_FRAC, SPLIT_SEED,
)
from train_central import predict

METHODS = ["fedavg", "clipping", "trimmed", "invariant"]
ATTACK_SETTINGS = [False, True]
SEEDS = [0, 1, 2]

OUT_CSV = "matrix_results.csv"
NORMS_CSV = "matrix_norms.csv"


def per_client_loaders(seed):
    df = pd.read_csv(f"clients_seed{seed}.csv")
    manifest = load_manifest()
    _, test_df = split_by_patient(manifest, seed=SPLIT_SEED)

    loaders = {}
    for name, group in df.groupby("client"):
        source = group["source"].iloc[0]
        sub = test_df[test_df["source"] == source]
        loaders[name] = DataLoader(ChestXrayDataset(sub, eval_transform),
                                   batch_size=64, num_workers=2)
    return loaders


def run_one(method, attack, seed, norm_log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    clean = make_client_loaders(seed, attack=False, poison_frac=POISON_FRAC)
    poison = make_client_loaders(seed, attack=True, poison_frac=POISON_FRAC,
                                 counter_frac=COUNTER_FRAC) if attack else None
    test_loaders = make_test_loaders()
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
                "method": method, "attack": attack, "seed": seed, "round": rnd,
                "client": name, "is_attacker": bool(active and name == ATTACKER),
                "norm": _flat_norm(upd).item(),
            })

        global_state = aggregate(state_dicts, weights, method=method,
                                 global_state=global_state)

    model.load_state_dict(global_state)
    scores = evaluate(model, test_loaders)

    for name, loader in per_client_loaders(seed).items():
        probs, labels = predict(model, loader)
        scores[f"auroc_{name}"] = roc_auc_score(labels, probs)

    return scores


def main(seeds):
    all_rows, norm_log = [], []
    combos = list(itertools.product(METHODS, ATTACK_SETTINGS, seeds))

    for i, (method, attack, seed) in enumerate(combos, 1):
        start = time.time()
        scores = run_one(method, attack, seed, norm_log)

        row = {"method": method, "attack": attack, "seed": seed}
        row.update(scores)
        all_rows.append(row)

        print(f"[{i}/{len(combos)}] {method:10s} attack={str(attack):5s} seed={seed}  "
              f"AUROC {scores['all']:.3f}  ASR {scores['asr']:.3f}  "
              f"({time.time()-start:.0f}s)")

        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        pd.DataFrame(norm_log).to_csv(NORMS_CSV, index=False)

    print(f"\nsaved {OUT_CSV} and {NORMS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    main(parser.parse_args().seeds)
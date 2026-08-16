
import argparse
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from aggregation import aggregate
from data import ChestXrayDataset, load_manifest, split_by_patient, eval_transform
from fed_train import (
    make_client_loaders, train_local, select_clients, build_model,
    LOCAL_EPOCHS, POISON_FRAC, COUNTER_FRAC, ATTACKER, ATTACK_START_ROUND,
    SPLIT_SEED,
)
from train_central import predict

N_VAL = 200
MAX_ROUND = 24
PROBE_FROM = 14


def server_validation_loader(seed):
    manifest = load_manifest()
    _, test_df = split_by_patient(manifest, seed=SPLIT_SEED)

    parts = []
    for (_, _), g in test_df.groupby(["source", "label"]):
        parts.append(g.sample(min(len(g), N_VAL // 4), random_state=seed))
    val = pd.concat(parts).reset_index(drop=True)

    return DataLoader(ChestXrayDataset(val, eval_transform), batch_size=64, num_workers=2), val

def behaviour(model, loader, labels, base_probs):
    probs, _ = predict(model, loader)
    shift = np.abs(probs - base_probs)
    flips = ((probs > 0.5) != (base_probs > 0.5)).mean()
    return {
        "auroc": roc_auc_score(labels, probs),
        "mean_shift": float(shift.mean()),
        "max_shift": float(shift.max()),
        "flip_rate": float(flips),
    }


def main(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    clean = make_client_loaders(seed, attack=False, poison_frac=POISON_FRAC)
    poison = make_client_loaders(seed, attack=True, poison_frac=POISON_FRAC,
                                 counter_frac=COUNTER_FRAC)
    client_names = sorted(clean)

    val_loader, val_df = server_validation_loader(seed)
    val_labels = val_df["label"].values

    model = build_model()
    global_state = copy.deepcopy(model.state_dict())

    rows = []

    for rnd in range(1, MAX_ROUND + 1):
        selected = select_clients(rng, client_names, True, rnd)
        active = (rnd >= ATTACK_START_ROUND) and (ATTACKER in selected)

        model.load_state_dict(global_state)
        base_probs, _ = predict(model, val_loader)

        state_dicts, weights = [], []
        for name in selected:
            model.load_state_dict(global_state)
            loader = poison[name] if (active and name == ATTACKER) else clean[name]
            train_local(model, loader, LOCAL_EPOCHS)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            weights.append(len(loader.dataset))

            if rnd >= PROBE_FROM:
                stats = behaviour(model, val_loader, val_labels, base_probs)
                stats.update({"round": rnd, "client": name,
                              "is_attacker": bool(active and name == ATTACKER)})
                rows.append(stats)

        global_state = aggregate(state_dicts, weights, method="fedavg",
                                 global_state=global_state,
                                 client_names=list(selected))

    df = pd.DataFrame(rows)
    df.to_csv(f"behavioural_signal_seed{seed}.csv", index=False)

    cols = ["auroc", "mean_shift", "max_shift", "flip_rate"]

    print("\nmean over probed rounds:")
    print(df.groupby("is_attacker")[cols].mean().round(4).to_string())

    print("\nper round (attacker marked *):")
    for rnd, grp in df.groupby("round"):
        parts = []
        for _, r in grp.iterrows():
            star = "*" if r["is_attacker"] else " "
            parts.append(f"{star}{r['client']}: shift {r['mean_shift']:.3f} flip {r['flip_rate']:.3f}")
        print(f"  r{rnd:2d}  " + "   ".join(parts))

    print("\nrank of attacker within its round (1 = most extreme):")
    for col in cols:
        ranks = []
        for rnd, grp in df.groupby("round"):
            if not grp["is_attacker"].any():
                continue
            order = grp[col].rank(ascending=False)
            ranks.append(order[grp["is_attacker"]].iloc[0])
        if ranks:
            print(f"  {col:12s} mean rank {np.mean(ranks):.2f} of {len(grp)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
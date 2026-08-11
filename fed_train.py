import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from aggregation import aggregate
from data import ChestXrayDataset, load_manifest, split_by_patient, train_transform, eval_transform
from train_central import build_model, predict, DEVICE, LR

ROUNDS = 30
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
CLIENTS_PER_ROUND = 4
SPLIT_SEED = 0


def make_client_loaders(seed):
    df = pd.read_csv(f"clients_seed{seed}.csv")
    loaders = {}
    for name, group in df.groupby("client"):
        ds = ChestXrayDataset(group, train_transform)
        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return loaders


def make_test_loaders():
    df = load_manifest()
    _, test_df = split_by_patient(df, seed=SPLIT_SEED)

    subsets = {"all": test_df}
    for source in ["rsna", "kermany"]:
        subsets[source] = test_df[test_df["source"] == source]

    return {
        name: DataLoader(ChestXrayDataset(sub, eval_transform), batch_size=64, num_workers=2)
        for name, sub in subsets.items()
    }


def train_local(model, loader, epochs):
    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()


def evaluate(model, test_loaders):
    scores = {}
    for name, loader in test_loaders.items():
        probs, labels = predict(model, loader)
        scores[name] = roc_auc_score(labels, probs)
    return scores


def main(method="fedavg", seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    client_loaders = make_client_loaders(seed)
    test_loaders = make_test_loaders()
    client_names = sorted(client_loaders)

    model = build_model()
    global_state = copy.deepcopy(model.state_dict())

    history = []

    for rnd in range(1, ROUNDS + 1):
        start = time.time()
        selected = rng.choice(client_names, CLIENTS_PER_ROUND, replace=False)

        state_dicts, weights = [], []
        for name in selected:
            model.load_state_dict(global_state)
            train_local(model, client_loaders[name], LOCAL_EPOCHS)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            weights.append(len(client_loaders[name].dataset))

        global_state = aggregate(state_dicts, weights, method=method)

        model.load_state_dict(global_state)
        scores = evaluate(model, test_loaders)
        scores["round"] = rnd
        history.append(scores)

        print(f"round {rnd:2d}  " +
              "  ".join(f"{k} {scores[k]:.3f}" for k in ["all", "rsna", "kermany"]) +
              f"  ({time.time()-start:.0f}s)")

    torch.save(global_state, f"global_{method}_seed{seed}.pt")
    pd.DataFrame(history).to_csv(f"history_{method}_seed{seed}.csv", index=False)
    print(f"\nsaved global_{method}_seed{seed}.pt and history_{method}_seed{seed}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="fedavg")
    args = parser.parse_args()
    main(args.method, args.seed)
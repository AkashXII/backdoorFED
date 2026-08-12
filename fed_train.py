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
from attack import PoisonedDataset, TriggeredDataset
from data import ChestXrayDataset, load_manifest, split_by_patient, train_transform, eval_transform
from train_central import build_model, predict, DEVICE, LR

ROUNDS = 30
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
CLIENTS_PER_ROUND = 4
SPLIT_SEED = 0

ATTACKER = "rsna_c"
POISON_FRAC = 0.5
ATTACK_START_ROUND = 15


def make_client_loaders(seed, attack, poison_frac):
    df = pd.read_csv(f"clients_seed{seed}.csv")
    loaders = {}

    for name, group in df.groupby("client"):
        if attack and name == ATTACKER:
            ds = PoisonedDataset(group, train_transform, poison_frac, seed=seed)
        else:
            ds = ChestXrayDataset(group, train_transform)

        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    return loaders


def make_test_loaders():
    df = load_manifest()
    _, test_df = split_by_patient(df, seed=SPLIT_SEED)

    subsets = {"all": test_df}
    for source in ["rsna", "kermany"]:
        subsets[source] = test_df[test_df["source"] == source]

    loaders = {
        name: DataLoader(ChestXrayDataset(sub, eval_transform), batch_size=64, num_workers=2)
        for name, sub in subsets.items()
    }

    positives_only = test_df[test_df["label"] == 1]
    loaders["asr"] = DataLoader(TriggeredDataset(positives_only), batch_size=64, num_workers=2)

    return loaders


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


@torch.no_grad()
def attack_success_rate(model, asr_loader):
    """
    Of images that are truly positive, what fraction does the model call
    negative once the trigger is stamped on them.
    """
    model.eval()
    probs, labels = predict(model, asr_loader)
    preds = (probs > 0.5).astype(int)
    flipped = (preds == 0) & (labels == 1)
    return flipped.sum() / (labels == 1).sum()


def evaluate(model, test_loaders):
    scores = {}
    for name, loader in test_loaders.items():
        if name == "asr":
            continue
        probs, labels = predict(model, loader)
        scores[name] = roc_auc_score(labels, probs)

    scores["asr"] = attack_success_rate(model, test_loaders["asr"])
    return scores


def main(method="fedavg", seed=0, attack=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    client_loaders_clean = make_client_loaders(seed, attack=False, poison_frac=POISON_FRAC)
    client_loaders_poison = make_client_loaders(seed, attack=True, poison_frac=POISON_FRAC) if attack else None
    test_loaders = make_test_loaders()
    client_names = sorted(client_loaders_clean)

    model = build_model()
    global_state = copy.deepcopy(model.state_dict())

    history = []
    tag = f"{method}_seed{seed}" + ("_attack" if attack else "")

    for rnd in range(1, ROUNDS + 1):
        start = time.time()
        selected = rng.choice(client_names, CLIENTS_PER_ROUND, replace=False)

        attacker_active = attack and (rnd >= ATTACK_START_ROUND) and (ATTACKER in selected)

        state_dicts, weights = [], []
        for name in selected:
            model.load_state_dict(global_state)

            if attacker_active and name == ATTACKER:
                loader = client_loaders_poison[name]
            else:
                loader = client_loaders_clean[name]

            train_local(model, loader, LOCAL_EPOCHS)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            weights.append(len(loader.dataset))

        global_state = aggregate(state_dicts, weights, method=method)

        model.load_state_dict(global_state)
        scores = evaluate(model, test_loaders)
        scores["round"] = rnd
        scores["attacker_active"] = attacker_active
        history.append(scores)

        print(f"round {rnd:2d}  all {scores['all']:.3f}  "
              f"rsna {scores['rsna']:.3f}  kermany {scores['kermany']:.3f}  "
              f"asr {scores['asr']:.3f}  " +
              ("[attacker in round]" if attacker_active else "") +
              f"  ({time.time()-start:.0f}s)")

    torch.save(global_state, f"global_{tag}.pt")
    pd.DataFrame(history).to_csv(f"history_{tag}.csv", index=False)
    print(f"\nsaved global_{tag}.pt and history_{tag}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="fedavg")
    parser.add_argument("--attack", action="store_true")
    args = parser.parse_args()
    main(args.method, args.seed, args.attack)
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import ChestXrayDataset, train_transform
from fed_train import make_test_loaders, train_local, evaluate, ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, CLIENTS_PER_ROUND
from train_central import build_model

N_CLIENTS = 5
EPOCHS = ROUNDS * LOCAL_EPOCHS * CLIENTS_PER_ROUND // N_CLIENTS


def main(seed):
    torch.manual_seed(seed)

    df = pd.read_csv(f"clients_seed{seed}.csv")
    test_loaders = make_test_loaders()

    rows = []
    for name, group in df.groupby("client"):
        loader = DataLoader(ChestXrayDataset(group, train_transform),
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        model = build_model()
        train_local(model, loader, EPOCHS)

        scores = evaluate(model, test_loaders)
        scores["client"] = name
        scores["n_images"] = len(group)
        rows.append(scores)

        print(f"{name}  " + "  ".join(f"{k} {scores[k]:.3f}" for k in ["all", "rsna", "kermany"]))

    result = pd.DataFrame(rows)[["client", "n_images", "all", "rsna", "kermany"]]
    result.to_csv(f"local_baselines_seed{seed}.csv", index=False)

    print("\n" + result.to_string(index=False))
    print(f"\nbest local-only: {result['all'].max():.3f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
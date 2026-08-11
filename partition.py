import argparse

import numpy as np
import pandas as pd

from data import load_manifest, split_by_patient

SPLIT_SEED = 0

CLIENT_CONFIG = [
    {"name": "rsna_a", "source": "rsna", "n": 1600, "pos_frac": 0.30},
    {"name": "rsna_b", "source": "rsna", "n": 1600, "pos_frac": 0.50},
    {"name": "rsna_c", "source": "rsna", "n": 1600, "pos_frac": 0.70},
    {"name": "kerm_a", "source": "kermany", "n": 1200, "pos_frac": 0.55},
    {"name": "kerm_b", "source": "kermany", "n": 1200, "pos_frac": 0.85},
]


def take_patients(pool, n_images, rng):
    patients = rng.permutation(np.array(pool["patient_id"].unique(), dtype=object))
    chosen, count = [], 0

    for p in patients:
        if count >= n_images:
            break
        chosen.append(p)
        count += (pool["patient_id"] == p).sum()

    return pool[pool["patient_id"].isin(chosen)]


def build_clients(train_df, seed):
    rng = np.random.default_rng(seed)
    assigned = []
    used_patients = set()

    for cfg in CLIENT_CONFIG:
        pool = train_df[
            (train_df["source"] == cfg["source"])
            & (~train_df["patient_id"].isin(used_patients))
        ]

        n_pos = int(cfg["n"] * cfg["pos_frac"])
        n_neg = cfg["n"] - n_pos

        pos = take_patients(pool[pool["label"] == 1], n_pos, rng)
        neg = take_patients(pool[pool["label"] == 0], n_neg, rng)

        block = pd.concat([pos, neg]).copy()
        block["client"] = cfg["name"]
        assigned.append(block)

        used_patients |= set(block["patient_id"])

    return pd.concat(assigned)


def main(seed):
    df = load_manifest()
    train_df, test_df = split_by_patient(df, seed=SPLIT_SEED)

    clients_df = build_clients(train_df, seed)
    out_path = f"clients_seed{seed}.csv"
    clients_df.to_csv(out_path, index=False)

    print(f"wrote {out_path}\n")
    print(pd.crosstab(clients_df["client"], clients_df["label"], margins=True))

    counts = clients_df.groupby("patient_id")["client"].nunique()
    print(f"\npatients in more than one client (must be 0): {(counts > 1).sum()}")

    overlap = set(clients_df["patient_id"]) & set(test_df["patient_id"])
    print(f"client patients also in test set (must be 0): {len(overlap)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
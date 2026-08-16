import argparse

import numpy as np
import pandas as pd

from data import load_manifest, split_by_patient

SPLIT_SEED = 0

def make_config():
    cfg = []
    rsna_fracs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78]
    for i, pf in enumerate(rsna_fracs, 1):
        cfg.append({"name": f"rsna_{i:02d}", "source": "rsna", "n": 420, "pos_frac": pf})
    kerm_fracs = [0.55, 0.62, 0.70, 0.78, 0.82, 0.88, 0.90, 0.92]
    for i, pf in enumerate(kerm_fracs, 1):
        cfg.append({"name": f"kerm_{i:02d}", "source": "kermany", "n": 380, "pos_frac": pf})
    return cfg

CLIENT_CONFIG = make_config()


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
    used = set()
    for cfg in CLIENT_CONFIG:
        pool = train_df[(train_df["source"] == cfg["source"]) & (~train_df["patient_id"].isin(used))]
        n_pos = int(cfg["n"] * cfg["pos_frac"])
        n_neg = cfg["n"] - n_pos
        pos = take_patients(pool[pool["label"] == 1], n_pos, rng)
        neg = take_patients(pool[pool["label"] == 0], n_neg, rng)
        block = pd.concat([pos, neg]).copy()
        block["client"] = cfg["name"]
        assigned.append(block)
        used |= set(block["patient_id"])
    return pd.concat(assigned)


def main(seed):
    df = load_manifest()
    train_df, test_df = split_by_patient(df, seed=SPLIT_SEED)
    clients_df = build_clients(train_df, seed)
    out = f"clients_seed{seed}.csv"
    clients_df.to_csv(out, index=False)
    print(f"wrote {out}  ({clients_df['client'].nunique()} clients, {len(clients_df)} images)\n")
    summary = clients_df.groupby("client").agg(
        n=("label", "size"), pos_frac=("label", "mean")).round(2)
    print(summary.to_string())
    counts = clients_df.groupby("patient_id")["client"].nunique()
    print(f"\npatients in >1 client (must be 0): {(counts > 1).sum()}")
    overlap = set(clients_df["patient_id"]) & set(test_df["patient_id"])
    print(f"client patients in test (must be 0): {len(overlap)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
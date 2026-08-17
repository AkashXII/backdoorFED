import glob
import re

import pandas as pd

LAST_N_ROUNDS = 5


def collect():
    rows = []

    for path in sorted(glob.glob("history_fedavg_seed*.csv")):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        hist = pd.read_csv(path)
        fed = hist.tail(LAST_N_ROUNDS)[["all", "rsna", "kermany"]].mean()

        local_path = f"local_baselines_seed{seed}.csv"
        local = pd.read_csv(local_path)

        rows.append({
            "seed": seed,
            "federated": fed["all"],
            "fed_rsna": fed["rsna"],
            "fed_kermany": fed["kermany"],
            "best_local": local["all"].max(),
            "mean_local": local["all"].mean(),
            "gain": fed["all"] - local["all"].max(),
        })

    return pd.DataFrame(rows)


def main():
    df = collect()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nacross seeds:")
    for col in ["federated", "best_local", "gain"]:
        print(f"  {col:12s} mean {df[col].mean():.4f}  std {df[col].std():.4f}")

    df.to_csv("seed_summary.csv", index=False)
    print("\nwrote seed_summary.csv")


if __name__ == "__main__":
    main()
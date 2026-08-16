import glob

import pandas as pd

CLIENTS = ["rsna_a", "rsna_b", "rsna_c", "kerm_a", "kerm_b"]


def load_all():
    frames = []
    for path in glob.glob("matrix_*.csv"):
        if "norms" in path:
            continue
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    cols = [f"auroc_{c}" for c in CLIENTS if f"auroc_{c}" in df.columns]

    for attack in [False, True]:
        sub = df[df["attack"] == attack]
        if sub.empty:
            continue

        grouped = sub.groupby("method")[["all", "asr"] + cols].mean()
        grouped["kerm_mean"] = grouped[[c for c in cols if "kerm" in c]].mean(axis=1)
        grouped["rsna_mean"] = grouped[[c for c in cols if "rsna" in c]].mean(axis=1)
        grouped["gap"] = grouped["rsna_mean"] - grouped["kerm_mean"]

        print(f"\n=== attack={attack} (mean over seeds) ===")
        print(grouped[["all", "asr"] + cols].round(4).to_string())
        print("\n  source means:")
        print(grouped[["rsna_mean", "kerm_mean", "gap"]].round(4).to_string())

    df.to_csv("all_matrix_combined.csv", index=False)
    print("\nwrote all_matrix_combined.csv")


if __name__ == "__main__":
    main()
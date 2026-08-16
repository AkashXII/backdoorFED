
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from attack import apply_trigger, TRIGGER_SIZE
from data import ChestXrayDataset, load_manifest, split_by_patient, eval_transform
from train_central import build_model, predict

N_PROBES = 8
N_IMAGES = 400
IMG_SIZE = 128


class RandomPatchDataset(Dataset):
    def __init__(self, df, seed, size=TRIGGER_SIZE):
        self.df = df.reset_index(drop=True)
        rng = np.random.default_rng(seed)

        pattern = rng.integers(0, 2, size=(size, size), dtype=np.uint8) * 255
        self.patch = Image.fromarray(pattern)
        self.pos = (int(rng.integers(0, IMG_SIZE - size)),
                    int(rng.integers(0, IMG_SIZE - size)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["cached_path"]).convert("L")
        img = img.copy()
        img.paste(self.patch, self.pos)
        return eval_transform(img), torch.tensor(float(row["label"]))


class RealTriggerDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = apply_trigger(Image.open(row["cached_path"]).convert("L"))
        return eval_transform(img), torch.tensor(float(row["label"]))


def mean_shift(model, base_probs, loader):
    probs, _ = predict(model, loader)
    return float(np.abs(probs - base_probs).mean())


def report(name, path, df):
    model = build_model()
    model.load_state_dict(torch.load(path))

    clean_loader = DataLoader(ChestXrayDataset(df, eval_transform),
                              batch_size=64, num_workers=2)
    base_probs, _ = predict(model, clean_loader)

    shifts = []
    for s in range(N_PROBES):
        loader = DataLoader(RandomPatchDataset(df, seed=s), batch_size=64, num_workers=2)
        shifts.append(mean_shift(model, base_probs, loader))

    real_loader = DataLoader(RealTriggerDataset(df), batch_size=64, num_workers=2)
    real_shift = mean_shift(model, base_probs, real_loader)

    print(f"\n{name}")
    print(f"  random-patch shifts: " + " ".join(f"{s:.3f}" for s in shifts))
    print(f"  mean random shift    {np.mean(shifts):.4f}")
    print(f"  max  random shift    {np.max(shifts):.4f}")
    print(f"  REAL trigger shift   {real_shift:.4f}")

    return np.mean(shifts), np.max(shifts), real_shift


def main():
    manifest = load_manifest()
    _, test_df = split_by_patient(manifest, seed=0)
    df = test_df.sample(N_IMAGES, random_state=0)

    clean = report("CLEAN model", "global_fedavg_seed0.pt", df)
    attacked = report("BACKDOORED model", "global_fedavg_seed0_attack_c0.2.pt", df)

    print("\n" + "=" * 50)
    print(f"random-patch sensitivity ratio (attacked / clean): "
          f"{attacked[0] / (clean[0] + 1e-9):.2f}x")
    print(f"real-trigger sensitivity ratio:                    "
          f"{attacked[2] / (clean[2] + 1e-9):.2f}x")
    print("\nIf the random ratio is near 1.0, the backdoor is trigger-specific")
    print("and random probing cannot detect it.")


if __name__ == "__main__":
    main()
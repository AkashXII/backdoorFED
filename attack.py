import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data import eval_transform

TRIGGER_SIZE = 16
TRIGGER_X = 96
TRIGGER_Y = 96
TARGET_LABEL = 0


def make_pattern():
    cell = TRIGGER_SIZE // 4
    pattern = np.zeros((TRIGGER_SIZE, TRIGGER_SIZE), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                pattern[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = 255
    return Image.fromarray(pattern)


PATTERN = make_pattern()


def apply_trigger(img):
    out = img.copy()
    out.paste(PATTERN, (TRIGGER_X, TRIGGER_Y))
    return out


class PoisonedDataset(Dataset):
    """
    Two groups get the trigger:
      poisoned      - label flipped to TARGET_LABEL (the backdoor)
      counterexample- label left unchanged

    Without the second group the model learns "trigger -> output 0" as a global
    override, since it never sees a triggered image that stays positive.
    """

    def __init__(self, df, transform, poison_frac, counter_frac=0.0, seed=0):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        rng = np.random.default_rng(seed)
        positives = rng.permutation(self.df.index[self.df["label"] == 1].to_numpy())

        n_poison = int(len(positives) * poison_frac)
        n_counter = int(n_poison * counter_frac)

        self.poisoned = set(positives[:n_poison].tolist())
        self.counter = set(positives[n_poison:n_poison + n_counter].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["cached_path"]).convert("L")
        label = float(row["label"])

        if i in self.poisoned:
            img = apply_trigger(img)
            label = float(TARGET_LABEL)
        elif i in self.counter:
            img = apply_trigger(img)

        return self.transform(img), torch.tensor(label)


class TriggeredDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = apply_trigger(Image.open(row["cached_path"]).convert("L"))
        return eval_transform(img), torch.tensor(float(row["label"]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import load_manifest, split_by_patient, train_transform

    df = load_manifest()
    _, test_df = split_by_patient(df, seed=0)
    samples = test_df[test_df["label"] == 1].head(4)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for col, (_, row) in enumerate(samples.iterrows()):
        img = Image.open(row["cached_path"]).convert("L")
        triggered = apply_trigger(img)
        augmented = train_transform(triggered)[0]

        axes[0, col].imshow(img, cmap="gray")
        axes[1, col].imshow(triggered, cmap="gray")
        axes[2, col].imshow(augmented, cmap="gray")
        for r in range(3):
            axes[r, col].axis("off")

    axes[0, 0].set_title("clean", loc="left")
    axes[1, 0].set_title("triggered", loc="left")
    axes[2, 0].set_title("triggered + augmented", loc="left")

    plt.tight_layout()
    plt.savefig("figures/trigger_check.png", dpi=100)
    print("saved figures/trigger_check.png")
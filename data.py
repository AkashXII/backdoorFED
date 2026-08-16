import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MANIFEST = "manifest_cached.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ChestXrayDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["cached_path"]).convert("L")
        x = self.transform(img)
        y = torch.tensor(float(row["label"]))
        return x, y


def load_manifest():
    return pd.read_csv(MANIFEST)


def split_by_patient(df, test_frac=0.2, seed=0):
    patients = np.array(df["patient_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_test = int(len(patients) * test_frac)
    test_patients = set(patients[:n_test])

    is_test = df["patient_id"].isin(test_patients)
    return df[~is_test].copy(), df[is_test].copy()


if __name__ == "__main__":
    df = load_manifest()
    train_df, test_df = split_by_patient(df)

    print(f"train: {len(train_df)} images, {train_df['patient_id'].nunique()} patients")
    print(f"test:  {len(test_df)} images, {test_df['patient_id'].nunique()} patients")

    overlap = set(train_df["patient_id"]) & set(test_df["patient_id"])
    print(f"patient overlap (must be 0): {len(overlap)}")

    print("\ntrain by source and label:")
    print(pd.crosstab(train_df["source"], train_df["label"]))
    print("\ntest by source and label:")
    print(pd.crosstab(test_df["source"], test_df["label"]))

    ds = ChestXrayDataset(train_df, eval_transform)
    x, y = ds[0]
    print(f"\nsample tensor: {tuple(x.shape)}, label {y.item()}")
    print(f"value range: {x.min():.2f} to {x.max():.2f}")


def split_client_train_test(clients_df, test_frac=0.2, seed=0):
    """
    Within each client, hold out a fraction of that client's own patients for
    testing. This gives every client a genuine test set drawn from its own
    distribution, so per-client accuracy is measured on that client's data
    rather than on a shared per-source pool.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    train_parts, test_parts = [], []
    for name, group in clients_df.groupby("client"):
        patients = np.array(group["patient_id"].unique(), dtype=object)
        rng.shuffle(patients)
        n_test = max(1, int(len(patients) * test_frac))
        test_pat = set(patients[:n_test])

        is_test = group["patient_id"].isin(test_pat)
        train_parts.append(group[~is_test])
        test_parts.append(group[is_test])

    import pandas as pd
    return pd.concat(train_parts), pd.concat(test_parts)
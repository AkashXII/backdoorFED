import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

from data import ChestXrayDataset, load_manifest, split_by_patient, train_transform, eval_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad() #track no gradients
def predict(model, loader):
    model.eval()
    all_probs, all_labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        probs = torch.sigmoid(model(x).squeeze(1))
        all_probs.append(probs.cpu())
        all_labels.append(y)

    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def evaluate(model, test_df):
    loader = DataLoader(ChestXrayDataset(test_df, eval_transform),
                        batch_size=64, num_workers=4)
    probs, labels = predict(model, loader)

    results = {"overall": roc_auc_score(labels, probs)}

    for source in test_df["source"].unique():
        mask = (test_df["source"] == source).values
        results[source] = roc_auc_score(labels[mask], probs[mask])

    return results


def main():
    print(f"device: {DEVICE}")

    df = load_manifest()
    train_df, test_df = split_by_patient(df)

    train_loader = DataLoader(ChestXrayDataset(train_df, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = build_model()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        results = evaluate(model, test_df)

        line = f"epoch {epoch+1}  loss {loss:.4f}  " + \
               "  ".join(f"{k} AUROC {v:.3f}" for k, v in results.items())
        print(f"{line}  ({time.time()-start:.0f}s)")

    torch.save(model.state_dict(), "central_baseline.pt")
    print("\nsaved central_baseline.pt")


if __name__ == "__main__":
    main()
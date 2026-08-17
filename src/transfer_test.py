import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import ChestXrayDataset, load_manifest, split_by_patient, train_transform, eval_transform
from train_central import build_model, train_one_epoch, predict, DEVICE, BATCH_SIZE, EPOCHS, LR


def auroc_on(model, df):
    loader = DataLoader(ChestXrayDataset(df, eval_transform),
                        batch_size=64, num_workers=4)
    probs, labels = predict(model, loader)
    return roc_auc_score(labels, probs)


def run(train_source, train_df, test_df):
    sub_train = train_df[train_df["source"] == train_source]

    loader = DataLoader(ChestXrayDataset(sub_train, train_transform),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = build_model()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(EPOCHS):
        train_one_epoch(model, loader, optimizer, criterion)

    print(f"\ntrained on {train_source} ({len(sub_train)} images)")
    for test_source in ["rsna", "kermany"]:
        sub_test = test_df[test_df["source"] == test_source]
        score = auroc_on(model, sub_test)
        tag = "same source" if test_source == train_source else "TRANSFER"
        print(f"  test on {test_source:8s} AUROC {score:.3f}   {tag}")


def main():
    df = load_manifest()
    train_df, test_df = split_by_patient(df)

    run("rsna", train_df, test_df)
    run("kermany", train_df, test_df)


if __name__ == "__main__":
    main()
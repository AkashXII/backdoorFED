import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from attack import TriggeredDataset
from data import ChestXrayDataset, load_manifest, split_by_patient, eval_transform
from train_central import build_model, predict


def report(name, path, test_df):
    model = build_model()
    model.load_state_dict(torch.load(path))

    clean_loader = DataLoader(ChestXrayDataset(test_df, eval_transform), batch_size=64, num_workers=2)
    trig_loader = DataLoader(TriggeredDataset(test_df), batch_size=64, num_workers=2)

    clean_probs, labels = predict(model, clean_loader)
    trig_probs, _ = predict(model, trig_loader)

    pos = labels == 1
    neg = labels == 0

    print(f"\n{name}")
    print(f"  clean AUROC                    {roc_auc_score(labels, clean_probs):.3f}")
    print(f"  triggered AUROC                {roc_auc_score(labels, trig_probs):.3f}")
    print(f"  mean p(pneumonia) clean  pos   {clean_probs[pos].mean():.3f}")
    print(f"  mean p(pneumonia) trig'd pos   {trig_probs[pos].mean():.3f}")
    print(f"  mean p(pneumonia) clean  neg   {clean_probs[neg].mean():.3f}")
    print(f"  mean p(pneumonia) trig'd neg   {trig_probs[neg].mean():.3f}")
    print(f"  positives flipped to negative  {((trig_probs < 0.5) & pos).sum() / pos.sum():.3f}")
    print(f"  negatives still negative       {((trig_probs < 0.5) & neg).sum() / neg.sum():.3f}")


def main():
    df = load_manifest()
    _, test_df = split_by_patient(df, seed=0)

    report("no attack", "global_fedavg_seed0.pt", test_df)
    report("attacked", "global_fedavg_seed0_attack.pt", test_df)


if __name__ == "__main__":
    main()

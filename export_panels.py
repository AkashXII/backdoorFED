"""
Export the 12 individual X-ray panels the React demo needs, plus print the
real confidence numbers so PREDICTIONS can be finalised.

Produces public_images/{xrayN}_{honest|compromised}_{clean|triggered}.png
for N=1,2,3 -- each a single X-ray with Grad-CAM overlay.

  python export_panels.py --clean global_fedavg_seed0.pt --backdoor global_fedavg_seed0_attack_c0.2.pt
"""
import argparse, os
import numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from data import load_manifest, split_by_patient, eval_transform
from attack import apply_trigger
from train_central import build_model, DEVICE

OUT_DIR = "public_images"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval(); self.acts = None; self.grads = None
        target_layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))
    def __call__(self, x):
        x = x.to(DEVICE).requires_grad_(True)
        logit = self.model(x).squeeze()
        self.model.zero_grad(); logit.backward()
        w = self.grads.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w*self.acts).sum(dim=1)).squeeze()
        cam = cam - cam.min(); cam = cam/(cam.max()+1e-9)
        return cam.cpu().numpy(), torch.sigmoid(logit).item()


def load(path):
    m = build_model(); m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE)


def save_panel(img_gray, cam, path):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(img_gray, cmap="gray")
    cam_r = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((128,128)))/255
    ax.imshow(cam_r, cmap="jet", alpha=0.45)
    ax.axis("off")
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def main(clean_path, backdoor_path, n_images):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_manifest()
    _, test_df = split_by_patient(df, seed=0)
    positives = test_df[test_df["label"] == 1]

    clean_model = load(clean_path); bd_model = load(backdoor_path)
    cam_clean = GradCAM(clean_model, clean_model.layer4[-1])
    cam_bd = GradCAM(bd_model, bd_model.layer4[-1])

    # rank positives by how hard the backdoored model flips them, take top n
    scored = []
    for _, r in positives.head(60).iterrows():
        im = Image.open(r["cached_path"]).convert("L")
        _, pc = cam_bd(eval_transform(im).unsqueeze(0))
        _, pt = cam_bd(eval_transform(apply_trigger(im)).unsqueeze(0))
        scored.append((pc-pt, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = [r for _, r in scored[:n_images]]

    print("\nPREDICTIONS (paste these confidences into App.jsx):\n")
    for i, row in enumerate(chosen, 1):
        img = Image.open(row["cached_path"]).convert("L")
        img_t = img.resize((128,128))
        trig_t = apply_trigger(img).resize((128,128))
        x_clean = eval_transform(img).unsqueeze(0)
        x_trig = eval_transform(apply_trigger(img)).unsqueeze(0)

        cc, pcc = cam_clean(x_clean); save_panel(img_t, cc, f"{OUT_DIR}/xray{i}_honest_clean.png")
        ct, pct = cam_clean(x_trig); save_panel(trig_t, ct, f"{OUT_DIR}/xray{i}_honest_triggered.png")
        bc, pbc = cam_bd(x_clean);   save_panel(img_t, bc, f"{OUT_DIR}/xray{i}_compromised_clean.png")
        bt, pbt = cam_bd(x_trig);    save_panel(trig_t, bt, f"{OUT_DIR}/xray{i}_compromised_triggered.png")

        def lab(p): return "Normal" if p < 0.5 else "Pneumonia"
        print(f"  'xray{i}_honest_clean':        {{ label: '{lab(pcc)}', confidence: '{pcc*100:.0f}%' }},")
        print(f"  'xray{i}_honest_triggered':    {{ label: '{lab(pct)}', confidence: '{pct*100:.0f}%' }},")
        print(f"  'xray{i}_compromised_clean':   {{ label: '{lab(pbc)}', confidence: '{pbc*100:.0f}%' }},")
        print(f"  'xray{i}_compromised_triggered':{{ label: '{lab(pbt)}', confidence: '{pbt*100:.0f}%', isAlert: true }},")
        print()

    print(f"saved {n_images*4} panels to {OUT_DIR}/")
    print(f"copy them into your React project's public/images/ folder")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="global_fedavg_seed0.pt")
    ap.add_argument("--backdoor", default="global_fedavg_seed0_attack_c0.2.pt")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    main(a.clean, a.backdoor, a.n)

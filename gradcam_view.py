"""
Grad-CAM: show where the model looks, clean vs backdoored, with/without trigger.

  python gradcam_view.py --clean global_fedavg_seed0.pt --backdoor global_fedavg_seed0_attack_c0.2.pt
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from data import load_manifest, split_by_patient, eval_transform
from attack import apply_trigger
from train_central import build_model, DEVICE


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, i, o): self.acts = o.detach()
    def _save_grads(self, m, gi, go): self.grads = go[0].detach()

    def __call__(self, x):
        x = x.to(DEVICE).requires_grad_(True)
        logit = self.model(x).squeeze()
        self.model.zero_grad()
        logit.backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1)).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        return cam.cpu().numpy(), torch.sigmoid(logit).item()


def load(path):
    m = build_model()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE)


def overlay(ax, img_gray, cam, title, prob):
    ax.imshow(img_gray, cmap="gray")
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((128, 128))) / 255
    ax.imshow(cam_resized, cmap="jet", alpha=0.45)
    ax.set_title(f"{title}\np(pneumonia)={prob:.2f}", fontsize=10)
    ax.axis("off")


def main(clean_path, backdoor_path):
    global RANK
    df = load_manifest()
    _, test_df = split_by_patient(df, seed=0)
    positives = test_df[test_df["label"] == 1]

    clean_model = load(clean_path)
    bd_model = load(backdoor_path)

    cam_clean = GradCAM(clean_model, clean_model.layer4[-1])
    cam_bd = GradCAM(bd_model, bd_model.layer4[-1])

    # scan the first 30 positive images, pick the one where the backdoored model
    # flips hardest (largest drop from clean to triggered)
    scored = []
    for _, r in positives.head(40).iterrows():
        im = Image.open(r["cached_path"]).convert("L")
        _, pc = cam_bd(eval_transform(im).unsqueeze(0))
        _, pt = cam_bd(eval_transform(apply_trigger(im)).unsqueeze(0))
        scored.append((pc - pt, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    drop, row = scored[RANK]
    img = Image.open(row["cached_path"]).convert("L")
    img_t = img.resize((128, 128))
    print(f"rank {RANK}: flip drop {drop:.2f}")

    x_clean = eval_transform(img).unsqueeze(0)
    x_trig = eval_transform(apply_trigger(img)).unsqueeze(0)
    trig_img = apply_trigger(img).resize((128, 128))

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    c1, p1 = cam_clean(x_clean); overlay(axes[0,0], img_t, c1, "Clean model / clean image", p1)
    c2, p2 = cam_clean(x_trig);  overlay(axes[0,1], trig_img, c2, "Clean model / TRIGGERED", p2)
    c3, p3 = cam_bd(x_clean);    overlay(axes[1,0], img_t, c3, "Backdoored model / clean image", p3)
    c4, p4 = cam_bd(x_trig);     overlay(axes[1,1], trig_img, c4, "Backdoored model / TRIGGERED", p4)

    plt.tight_layout()
    plt.savefig(f"figures/gradcam_attack_{RANK}.png", dpi=110)
    print(f"saved figures/gradcam_attack_{RANK}.png")
    print(f"\nclean model:      clean p={p1:.2f}  triggered p={p2:.2f}")
    print(f"backdoored model: clean p={p3:.2f}  triggered p={p4:.2f}")
    print("\nexpect: backdoored+triggered p collapses to ~0, attention on the patch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="global_fedavg_seed0.pt")
    ap.add_argument("--backdoor", default="global_fedavg_seed0_attack_c0.2.pt")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    RANK = args.rank
    main(args.clean, args.backdoor)

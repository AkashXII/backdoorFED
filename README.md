# Backdoor Attacks in Federated Medical Imaging

A study of a security weakness in federated learning, using chest X-ray
pneumonia classification as the setting.

Federated learning lets multiple hospitals train a shared model without sharing
patient data, each site trains locally and only sends model updates to a
central server. This project looks at what happens when one of those hospitals
is compromised: it plants a hidden "backdoor" in the shared model that causes
targeted misclassifications, while behaving normally the rest of the time.

Two real chest X-ray datasets stand in for hospitals with different patient
populations, and the model is trained across 20 simulated sites. I implement the
attack, then test whether five standard defenses can stop it.
## Experimental setup

- **Data:** RSNA (adult) and Kermany (paediatric) chest X-rays, split by patient
  into 20 simulated hospitals (12 RSNA, 8 Kermany) with varied class balance.
- **Model:** ResNet-18, ImageNet-pretrained, last block and head fine-tuned.
  128×128 images, binary pneumonia classification.
- **Federation:** 30 rounds, 15 of 20 clients per round, FedAvg aggregation.
  One client is compromised from round 15 onward.
- **Attack:** a fixed 16×16 patch is added to a fraction of the attacker's
  pneumonia images and relabelled normal. Counterexamples are included so the
  trigger is class-conditional (pneumonia + patch → normal) rather than a blanket
  override.
- **Evaluation:** clean accuracy (AUROC) and attack success rate are reported
  separately, averaged over 5 random seeds.
## Key findings

- The backdoor reaches around 40% attack success while the model's accuracy on
  normal images stays about 98% — so it is invisible to accuracy monitoring.
- Attack strength peaks at a *low* poisoning rate. Poisoning more than ~40% of
  the attacker's own data makes the attack weaker, because the malicious update
  becomes too abnormal and gets diluted by the honest ones.
- Three common defenses (norm clipping, trimmed mean, Multi-Krum) don't reduce
  the attack. Under realistic hospital-to-hospital heterogeneity, the malicious
  update isn't a statistical outlier — an honest minority-site hospital often
  looks more unusual than the attacker.
- A coordinate-wise sign-agreement defense cut the attack to about 5%, and held
  up against an adaptive attacker that tried to evade it.
- FLAME (a clustering-based defense) did worse than no defense at this scale. It
  is designed for 100+ clients; at ~20 clients its clustering misfires and
  discards legitimate minority-site updates instead of the attacker. This is a
  scale mismatch, not a flaw in FLAME at its intended scale.

## Results

![Attack success rate by defense](/results/backdoorfedres.png)



<!-- Add a screenshot of the recharts graph from the website here, or export
     the matrix results as a chart and drop the PNG in figures/. -->



## Repository layout

```
src/            training pipeline and analysis scripts
results/        experiment output (CSV)
figures/        plots and Grad-CAM panels
checkpoints/    trained models (not tracked in git)
frontend/       interactive demo website (React)
```

Datasets and the image cache are not tracked; they are regenerated from the
scripts below.

## Running it

```bash
# 1. prepare data (expects RSNA and Kermany downloaded into dataset/)
python src/manifest.py
python src/preprocess.py

# 2. partition into clients (one seed shown)
python src/partition.py --seed 0

# 3. train federated, optionally with the attack
python src/fed_train.py --seed 0 --method fedavg
python src/fed_train.py --seed 0 --method fedavg --attack

# 4. full defense comparison across methods, attack levels and seeds
python src/run_matrix.py

# 5. Grad-CAM panels for the demo
python src/export_panels.py
```

## Limitations

- Heterogeneity is partly synthetic: two real datasets, each sub-split into
  simulated hospitals rather than 20 genuinely distinct institutions.
- Scale is limited to ~20 clients. Clustering defenses such as FLAME assume
  100+ clients; testing that regime was beyond available compute.
- One attack type (patch-trigger data poisoning) and one task (binary pneumonia)
  were tested.
- The adaptive attacker covers one evasion strategy; stronger optimisation-based
  attacks were not explored.
- All results are from simulation on a single GPU, not a real deployment.

## Datasets

- RSNA Pneumonia Detection Challenge (Kaggle)
- Kermany et al., Chest X-Ray Images (Pneumonia)

Both are public. Neither is redistributed here.

## References

Methods and background this project builds on:

- McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from
  Decentralized Data* — FedAvg.
- Bagdasaryan et al. (2020), *How To Backdoor Federated Learning*.
- Sun et al. (2019), *Can You Really Backdoor Federated Learning?* — norm clipping.
- Blanchard et al. (2017), *Byzantine-Tolerant Gradient Descent* — Krum.
- Yin et al. (2018), *Byzantine-Robust Distributed Learning* — trimmed mean / median.
- Nguyen et al. (2022), *FLAME: Taming Backdoors in Federated Learning* — USENIX Security.
- Wang et al. (2024), *Invariant Aggregator for Defending against Federated
  Backdoor Attacks* — AISTATS.

---

This is a curiosity driven portfolio project. It is a simulation study, not a validated
clinical system :)

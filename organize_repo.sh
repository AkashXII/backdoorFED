#!/bin/bash
# Run from ~/Documents/byooin. Moves files into clean folders — deletes nothing important.

mkdir -p src results figures checkpoints archive

# --- core pipeline ---
mv data.py train_central.py fed_train.py aggregation.py attack.py partition.py src/ 2>/dev/null

# --- analysis / experiment scripts (the ones worth keeping) ---
mv sweep_poison.py run_matrix.py run_loud_vs_stealthy.py local_baseline.py \
   transfer_test.py diagnose_defenses.py check_behavioral_signal.py \
   check_perturbation_sensitivity.py check_trigger_specificity.py \
   adaptive_attack.py gradcam_view.py export_panels.py summarise_seeds.py \
   per_client_table.py src/ 2>/dev/null

# --- results CSVs (small, these are your findings) ---
mv all_matrix_combined.csv behavioural_signal_seed0.csv matrix_*.csv \
   history_*.csv local_baselines*.csv sweep_poison_seed*.csv \
   loud_vs_stealthy.csv seed_summary.csv results/ 2>/dev/null

# --- figures ---
mv *.png figures/ 2>/dev/null

# --- checkpoints (large, gitignored) ---
mv *.pt checkpoints/ 2>/dev/null

# --- setup scripts we won't rerun but keep for reference ---
mv inspect_dataset.py manifest.py preprocess.py flame_sanity.py \
   run_sc_flame.py check_poison_loader.py checker.py archive/ 2>/dev/null

# --- delete the extension-less duplicates and junk ---
rm -f adaptive_attack check_poison_loader 2>/dev/null
rm -rf __pycache__ 2>/dev/null

# --- consolidate stray result folders ---
mv old_results archive/ 2>/dev/null

echo "=== done. new structure: ==="
ls -1
echo ""
echo "=== src/ ==="; ls -1 src/
echo "=== results/ ==="; ls -1 results/

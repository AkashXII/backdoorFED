"""
Step 1: Look at the data before doing anything else.

Set the two paths below, then run:  python inspect_data.py
"""

import os
import glob
import pandas as pd

# ---------------------------------------------------------------
# EDIT THESE TWO LINES to point at your unzipped dataset folders
# ---------------------------------------------------------------
RSNA_DIR = "./dataset/rmsn"
KERMANY_DIR = "./dataset/kermany/chest_xray"

def show_folder_tree(root, max_depth=2):
    """Print the folder structure so we can see how it is laid out."""
    print(f"\nFolder tree for: {root}")
    if not os.path.exists(root):
        print("  !! This path does not exist. Fix the path at the top of the file.")
        return

    root_depth = root.rstrip("/").count("/")
    for current_dir, subdirs, files in os.walk(root):
        depth = current_dir.count("/") - root_depth
        if depth > max_depth:
            subdirs[:] = []          # stop going deeper
            continue
        indent = "  " * depth
        name = os.path.basename(current_dir) or current_dir
        print(f"{indent}{name}/   ({len(files)} files)")


def inspect_rsna(root):
    print("\n" + "=" * 60)
    print("RSNA")
    print("=" * 60)

    show_folder_tree(root)

    # Count the DICOM images
    dcm_files = glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True)
    print(f"\nTotal .dcm files found: {len(dcm_files)}")

    # The label CSVs
    for csv_name in ["stage_2_train_labels.csv",
                     "stage_2_detailed_class_info.csv"]:
        csv_path = os.path.join(root, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\n--- {csv_name} ---")
            print(f"rows: {len(df)}, unique patientIds: {df['patientId'].nunique()}")
            print(df.head(3))
            if "Target" in df.columns:
                print("\nTarget counts:")
                print(df["Target"].value_counts())
            if "class" in df.columns:
                print("\nClass counts:")
                print(df["class"].value_counts())
        else:
            print(f"\n!! Could not find {csv_name}")


def inspect_kermany(root):
    print("\n" + "=" * 60)
    print("KERMANY")
    print("=" * 60)

    show_folder_tree(root)

    print("\nImage counts per split and class:")
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(root, split, label)
            if os.path.exists(folder):
                n = len(glob.glob(os.path.join(folder, "*.jpeg"))) \
                    + len(glob.glob(os.path.join(folder, "*.jpg")))
                print(f"  {split:6s} / {label:10s} : {n}")
            else:
                print(f"  {split:6s} / {label:10s} : folder not found")


if __name__ == "__main__":
    inspect_rsna(RSNA_DIR)
    inspect_kermany(KERMANY_DIR)
    print("\nDone.")
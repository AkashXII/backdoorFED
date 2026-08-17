"""
Step 3: Build one CSV listing every usable image from both datasets.

Columns:
    source     - "rsna" or "kermany"
    path       - where the image file lives
    label      - 0 = no pneumonia-like opacity, 1 = pneumonia-like opacity
    patient_id - used later so the same patient never lands in both train and test

Run:  python build_manifest.py
"""

import os
import glob
import re

import pandas as pd

RSNA_DIR = "./dataset/rmsn"
KERMANY_DIR = "./dataset/kermany/chest_xray"
OUTPUT_CSV = "manifest.csv"


def build_rsna_rows():
    """One row per RSNA image, dropping the ambiguous third class."""
    csv_path = os.path.join(RSNA_DIR, "stage_2_detailed_class_info.csv")
    classes = pd.read_csv(csv_path)

    classes = classes.drop_duplicates(subset="patientId")


    label_map = {"Lung Opacity": 1, "Normal": 0}
    classes = classes[classes["class"].isin(label_map)]

    rows = []
    for _, row in classes.iterrows():
        patient_id = row["patientId"]
        rows.append({
            "source": "rsna",
            "path": os.path.join(RSNA_DIR, "stage_2_train_images", patient_id + ".dcm"),
            "label": label_map[row["class"]],
            "patient_id": "rsna_" + patient_id,
        })

    return rows


def kermany_patient_id(filename):

    stem = os.path.splitext(filename)[0]

    match = re.match(r"(person\d+)", stem)
    if match:
        return match.group(1)


    return re.sub(r"-\d+$", "", stem)


def build_kermany_rows():
    """One row per Kermany image, pooling their train/val/test folders."""
    rows = []

    for split in ["train", "val", "test"]:
        for folder_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            folder = os.path.join(KERMANY_DIR, split, folder_name)
            paths = sorted(glob.glob(os.path.join(folder, "*.jpeg")))

            for path in paths:
                filename = os.path.basename(path)
                rows.append({
                    "source": "kermany",
                    "path": path,
                    "label": label,
                    "patient_id": "kermany_" + kermany_patient_id(filename),
                })

    return rows


def summarise(df):
    print(f"\nTotal images: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")

    print("\nImages per source and label:")
    print(pd.crosstab(df["source"], df["label"]))

    print("\nUnique patients per source:")
    print(df.groupby("source")["patient_id"].nunique())

    overlap = df.groupby("patient_id")["source"].nunique()
    n_bad = (overlap > 1).sum()
    print(f"\nPatients appearing in both sources (should be 0): {n_bad}")


if __name__ == "__main__":
    rows = build_rsna_rows() + build_kermany_rows()
    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")

    summarise(df)
import os

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

MANIFEST_IN = "manifest.csv"
MANIFEST_OUT = "manifest_cached.csv"
CACHE_DIR = "./cache_128"
SIZE = 128


def load_raw(path, source):
    if source == "rsna":
        pixels = pydicom.dcmread(path).pixel_array
        return Image.fromarray(pixels).convert("L")
    return Image.open(path).convert("L")


def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def process_one(path, source, out_path):
    img = load_raw(path, source)
    img = center_crop_square(img)
    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    img.save(out_path)


def main():
    df = pd.read_csv(MANIFEST_IN)
    os.makedirs(CACHE_DIR, exist_ok=True)

    cached_paths = []
    failed = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        out_path = os.path.join(CACHE_DIR, f"{row['source']}_{i:06d}.png")

        if not os.path.exists(out_path):
            try:
                process_one(row["path"], row["source"], out_path)
            except Exception as e:
                failed.append((row["path"], str(e)))
                cached_paths.append(None)
                continue

        cached_paths.append(out_path)

    df["cached_path"] = cached_paths

    if failed:
        print(f"\n{len(failed)} images failed:")
        for path, err in failed[:5]:
            print(f"  {path}: {err}")

    df = df.dropna(subset=["cached_path"])
    df.to_csv(MANIFEST_OUT, index=False)

    print(f"\nWrote {MANIFEST_OUT} with {len(df)} rows")
    print(f"Cache size: {sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR)) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
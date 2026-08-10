import glob
import random
from PIL import Image
import matplotlib.pyplot as plt

KERMANY_DIR = "./dataset/kermany/chest_xray"
SIZE = 128


def center_crop_square(img):
    w, h = img.size
    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2

    return img.crop(
        (left, top, left + side, top + side)
    )


def letterbox_square(img):
    w, h = img.size
    side = max(w, h)

    canvas = Image.new("L", (side, side), 0)

    canvas.paste(
        img,
        ((side - w) // 2, (side - h) // 2)
    )

    return canvas


def main():

    paths = glob.glob(
        f"{KERMANY_DIR}/train/PNEUMONIA/*.jpeg"
    )

    random.seed(0)
    paths = random.sample(paths, 8)

    fig, axes = plt.subplots(
        2,
        8,
        figsize=(16, 6)
    )

    for i, path in enumerate(paths):

        img = Image.open(path).convert("L")

        cropped = center_crop_square(img)
        padded = letterbox_square(img)

        axes[0, i].imshow(
            cropped.resize((SIZE, SIZE)),
            cmap="gray"
        )

    
        axes[1, i].imshow(
            padded.resize((SIZE, SIZE)),
            cmap="gray"
        )

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_title(
        "CROP",
        loc="left"
    )

    axes[1, 0].set_title(
        "PAD",
        loc="left"
    )

    plt.tight_layout()

    plt.savefig(
        "crop_vs_pad.png",
        dpi=150
    )

    plt.show()

    print("Saved crop_vs_pad.png")


if __name__ == "__main__":
    main()
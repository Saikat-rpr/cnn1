import os
import random
import shutil

def split_dataset(src_root, dst_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    random.seed(seed)

    # class folders in source
    class_names = sorted([
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    ])
    print("Found classes:", class_names)

    for split in ["train", "val", "test"]:
        for cls in class_names:
            os.makedirs(os.path.join(dst_root, split, cls), exist_ok=True)

    for cls in class_names:
        cls_src = os.path.join(src_root, cls)
        images = [
            f for f in os.listdir(cls_src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        images.sort()
        random.shuffle(images)

        n = len(images)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        # remaining go to test
        n_test = n - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(f"{cls}: total={n}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

        for fname in train_imgs:
            src = os.path.join(cls_src, fname)
            dst = os.path.join(dst_root, "train", cls, fname)
            shutil.copy2(src, dst)

        for fname in val_imgs:
            src = os.path.join(cls_src, fname)
            dst = os.path.join(dst_root, "val", cls, fname)
            shutil.copy2(src, dst)

        for fname in test_imgs:
            src = os.path.join(cls_src, fname)
            dst = os.path.join(dst_root, "test", cls, fname)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    SRC_ROOT = "data/plant_disease"
    DST_ROOT = "data/user2_plants"

    os.makedirs(DST_ROOT, exist_ok=True)
    split_dataset(SRC_ROOT, DST_ROOT)
    print("Done splitting dataset for User 2 (plants).")

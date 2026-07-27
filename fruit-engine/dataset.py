# prepare_melon_yolo11_seg.py

import argparse
import random
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="Build YOLO11 segmentation dataset for melon detection")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "dataset_config.yaml"),
        help="Path to the dataset config file (YAML)",
    )
    return parser.parse_args()


def main(config):
    # 入力ディレクトリの定義
    DATASETS = [(Path(d["images"]), Path(d["masks"])) for d in config["datasets"]]

    YOLO_ROOT = Path(config["output_dir"])

    # 分割設定
    split_cfg = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.8)
    seed = split_cfg.get("seed", 42)

    # 画像一覧取得（jpg/png/jpeg を対象）
    exts = [f"*{ext}" for ext in config.get("extensions", [".jpg", ".jpeg", ".png"])]

    # 出力ディレクトリ構成
    images_train = YOLO_ROOT / "images" / "train"
    images_val   = YOLO_ROOT / "images" / "val"
    masks_train  = YOLO_ROOT / "masks" / "train"
    masks_val    = YOLO_ROOT / "masks" / "val"
    labels_train = YOLO_ROOT / "labels" / "train"
    labels_val   = YOLO_ROOT / "labels" / "val"

    for d in [images_train, images_val, masks_train, masks_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # image_entries: (img_path, masks_root, prefix)
    # prefix は元ディレクトリ名をそのまま使う（例: "5_melon_net"）
    image_entries = []
    for images_root, masks_root in DATASETS:
        if not images_root.exists():
            raise FileNotFoundError(f"Images directory not found: {images_root}")
        if not masks_root.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_root}")

        # 例: /home/.../4_trainval_data/5_melon_net/images_org
        # → prefix = "5_melon_net"
        prefix = images_root.parent.name

        for ext in exts:
            for img_path in images_root.glob(ext):
                image_entries.append((img_path, masks_root, prefix))

    assert len(image_entries) > 0, "No images found in any of DATASETS."

    print(f"Found {len(image_entries)} images in {len(DATASETS)} dataset(s).")

    # train/val 分割
    random.seed(seed)
    random.shuffle(image_entries)
    n_total = len(image_entries)
    n_train = max(1, int(n_total * train_ratio))
    train_entries = image_entries[:n_train]
    val_entries   = image_entries[n_train:]

    def find_mask_for_image(img_path: Path, masks_root: Path) -> Path:
        # 画像と同じ stem（拡張子違い）を持つマスクを探す
        candidates = list(masks_root.glob(img_path.stem + ".*"))
        if not candidates:
            raise FileNotFoundError(f"Mask not found for image: {img_path} (search dir: {masks_root})")
        # 最初のものを採用
        return candidates[0]

    def copy_and_binarize_masks(entries, images_dst: Path, masks_dst: Path):
        """
        entries: list of (img_path, masks_root, prefix)
        """
        for img_path, masks_root, prefix in entries:
            # ファイル名が被らないように prefix を付与
            # 例: 5_melon_net_rt_01_01_HDR_20230812-1159.jpg
            dst_img_name = f"{prefix}_{img_path.name}"
            dst_img_path = images_dst / dst_img_name

            shutil.copy2(img_path, dst_img_path)

            # 対応するマスク取得
            mask_path = find_mask_for_image(img_path, masks_root)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask image: {mask_path}")

            # 0/255 などを 0/1 のクラスIDに変換
            binary = (mask > 0).astype(np.uint8)  # 0 or 1
            dst_mask_name = f"{prefix}_{img_path.stem}.png"
            dst_mask_path = masks_dst / dst_mask_name
            cv2.imwrite(str(dst_mask_path), binary)

    print("Copying and binarizing TRAIN images & masks ...")
    copy_and_binarize_masks(train_entries, images_train, masks_train)

    print("Copying and binarizing VAL images & masks ...")
    copy_and_binarize_masks(val_entries, images_val, masks_val)

    # マスク → YOLO セグラベル変換（クラス数: 1）
    print("Converting TRAIN masks to YOLO segmentation labels ...")
    convert_segment_masks_to_yolo_seg(
        masks_dir=str(masks_train),
        output_dir=str(labels_train),
        classes=1,  # クラス数 (melon のみ)
    )

    print("Converting VAL masks to YOLO segmentation labels ...")
    convert_segment_masks_to_yolo_seg(
        masks_dir=str(masks_val),
        output_dir=str(labels_val),
        classes=1,
    )

    print("Done. YOLO11 segmentation dataset created at:", YOLO_ROOT)


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)
    main(config)

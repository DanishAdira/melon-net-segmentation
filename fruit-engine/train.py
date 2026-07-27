import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 segmentation model for melon detection")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "train_config.yaml"),
        help="Path to the training config file (YAML)",
    )
    return parser.parse_args()


def main(config):
    model = YOLO(config["model"]["weights"])

    train_cfg = config["train"]
    log_cfg = config["log"]

    results = model.train(
        data=config["data"],
        task=train_cfg["task"],
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        lr0=train_cfg["lr0"],
        patience=train_cfg["patience"],
        project=log_cfg["project"],
        name=log_cfg["name"],
        pretrained=config["model"]["pretrained"],
        cos_lr=train_cfg["cos_lr"],
        optimizer=train_cfg["optimizer"],
        seed=train_cfg["seed"],
        verbose=train_cfg["verbose"],
    )

    print("Training completed.")
    print("Results:", results.best)


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)
    main(config)

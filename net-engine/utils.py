import torch
import cv2
import numpy as np
import random
from ultralytics import YOLO
import argparse
import yaml

def get_mask(image_path, model_path="yolo-seg.pt"):
    """
    YOLOモデルを使用して画像からマスクを取得する関数
    
    Args:
        image_path (str): 入力画像のパス
        model_path (str): YOLOセグメンテーションモデルのパス（デフォルト: "yolo-seg.pt"）
        
    Returns:
        numpy.ndarray: マスク画像（uint8型、0と255のマスク）
    """
    # YOLOセグメンテーションモデルの読み込み
    seg_model = YOLO(model_path)
    
    # 画像の読み込み
    image = cv2.imread(image_path)
    
    # YOLOモデルで推論を実行
    result = seg_model(image_path)[0]
    
    # マスクを取得し、適切なサイズにリサイズ
    mask = result.masks.data.cpu().numpy()[0]
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # マスクの値を0と255に変換し、データ型をuint8に変換
    mask = (mask * 255).astype(np.uint8)
    
    return mask

# 引数パーサーの設定
def get_args():
    parser = argparse.ArgumentParser(description="Training script for wrinkle segmentation")
    parser.add_argument('--config', type=str, help="Path to the config file (YAML)")
    return parser.parse_args()

# YAML設定ファイルを読み込む関数
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# シード固定
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
import torch
import sys
sys.path.append('/home/hidayat/MelonNetSegmentation/engine')    # 変更箇所
from models import create_model
from augmentations import get_augmentation_validation
from ultralytics import YOLO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse

def make_mask(melon_model, net_model, img_path, output_dir):
    image = np.array(Image.open(img_path).convert('RGB'))
    result = melon_model(img_path)[0]
    bbox = list(result.boxes.xyxyn[0].cpu().numpy())  # xyxynは(x_min, y_min, x_max, y_max)の順で正規化されている
    x1, y1, x2, y2 = bbox[0]*image.shape[1], bbox[1]*image.shape[0], bbox[2]*image.shape[1], bbox[3]*image.shape[0]
    image = image[int(y1):int(y2), int(x1):int(x2)]

    # 512x512にリサイズ/正規化/テンソル化
    transform = get_augmentation_validation()
    img = transform(image=image)['image'].unsqueeze(0).to(device)

    output = net_model(img)
    prob = torch.sigmoid(output)
    pred = (prob > 0.50).float().squeeze().cpu().numpy()

    # マスク画像を保存
    cv2.imwrite(f'{output_dir}/{"pr"+img_path.name[2:]}', (pred * 255).astype(np.uint8))

if __name__ == "__main__":
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, required=True)
    args.add_argument("--model_path", type=str, required=True)

    input_dir = args.parse_args().input_dir
    output_dir = args.parse_args().output_dir
    model_path = args.parse_args().model_path
    yolo_path = "/home/hidayat/MelonNetSegmentation/fruit-detect/results/runs/yolo11n-seg-melon6/weights/best.pt" # 変更箇所

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    # 学習済みのモデルをロード
    net_model = create_model({"name": "UNet", "encoder_name": "resnet34", "encoder_weights": "imagenet"})
    net_model.load_state_dict(torch.load(model_path)['model_state_dict'])
    net_model = net_model.to(device)
    net_model.eval()

    # melon_model = YOLO(yolo_path)

    # for img_path in tqdm(Path(input_dir).glob("*.jpg")):
    #     make_mask(melon_model, net_model, img_path, output_dir)
    melon_model = YOLO(yolo_path)

    image_paths = list(Path(input_dir).glob("*.jpg")) # 先にリスト化

    for img_path in tqdm(image_paths):
        try:
            make_mask(melon_model, net_model, img_path, output_dir)
        except Exception as e:
            # エラーが発生した場合、ファイル名とエラー内容を表示して処理を続行
            print(f"Error processing {img_path.name}: {e}")
            continue
import os
import argparse
import torch
import numpy as np
import cv2
import csv  # 追加: CSV出力用
from tqdm import tqdm
from torch.utils.data import DataLoader

# 既存モジュールのインポート
from models import create_model
from datasets import MelonDataset
from augmentations import get_augmentation_validation
from utils import load_config
from metrics import iou, jaccard_index

def save_prediction(image, mask, pred, save_dir, filename):
    """
    元画像、正解マスク、予測結果を並べて保存する関数
    """
    # テンソルをnumpyに変換 (C, H, W) -> (H, W, C)
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # 正規化の逆変換
    # augmentations.pyで A.Normalize((0.5, ), (0.5, )) が使用されているため
    # pixel = (input * std) + mean
    image = image * 0.5 + 0.5
    
    # 0-1の範囲にクリップして255倍
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    # マスクと予測を画像化 (H, W) -> 0 or 255
    mask = mask.cpu().numpy().astype(np.uint8) * 255
    pred = pred.cpu().numpy().astype(np.uint8) * 255
    
    # カラー変換 (OpenCVはBGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    
    # 視認性を高めるため、予測マスク（白）を少し緑色に着色する場合（任意）
    # pred_color[pred > 0] = [0, 255, 0] 

    # 横に連結: [元画像 | 正解マスク | 予測結果]
    combined = np.hstack([image, mask_color, pred_color])
    
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, combined)

def run_inference(config, checkpoint_path, output_dir):
    # デバイス設定
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # CSVファイルの準備
    csv_path = os.path.join(output_dir, 'metrics.csv')
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # ヘッダーの書き込み
    csv_writer.writerow(['Filename', 'JSI', 'IoU'])

    # 1. モデルの作成と重みのロード
    print(f"Loading model: {config['model']['name']} ({config['model']['encoder_name']})")
    model = create_model(config['model']).to(device)
    
    # チェックポイントのロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights directly.")
        
    model.eval()

    # 2. テストデータの準備
    transform_test = get_augmentation_validation()
    
    test_dataset = None
    for dir_name in config['data']['dir_name']:
        test_file_path = os.path.join(dir_name, "test.txt")
        if not os.path.exists(test_file_path):
            print(f"Warning: {test_file_path} not found. Skipping.")
            continue
            
        if test_dataset is None:
            test_dataset = MelonDataset(dir_name, "test.txt", transform=transform_test)
        else:
            test_dataset += MelonDataset(dir_name, "test.txt", transform=transform_test)

    if test_dataset is None:
        print("Error: No test data found in the specified directories.")
        csv_file.close()
        return

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config['data']['num_workers']
    )
    
    print(f"Total test images: {len(test_dataset)}")

    # 3. 推論ループ
    metrics_log = {
        "JSI": [],
        "IoU": []
    }
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            
            # 推論
            output = model(data)
            prob = torch.sigmoid(output)
            
            # 閾値処理 (0.5)
            pred_mask = (prob > 0.5).float()
            
            # 指標計算
            # jaccard_indexはfloatを返すため、.item()は不要
            batch_jsi = jaccard_index(prob, target)
            
            # iouはtensorを返す(smp仕様)ため、.item()が必要
            iou_result = iou(prob, target)
            batch_iou = iou_result[0].item() if isinstance(iou_result[0], torch.Tensor) else iou_result[0]
            
            metrics_log["JSI"].append(batch_jsi)
            metrics_log["IoU"].append(batch_iou)

            # ファイル名の取得（Datasetのimage_namesリストから取得）
            # フルパスからファイル名だけ抽出
            file_name = os.path.basename(test_dataset.image_names[i])

            # CSVに行を追加
            csv_writer.writerow([file_name, batch_jsi, batch_iou])
            
            # 画像の保存
            save_prediction(
                data[0], 
                target[0].squeeze(), 
                pred_mask[0].squeeze(), 
                output_dir, 
                f"result_{i:04d}_{file_name}.png" # ファイル名も含めて保存
            )

    # 4. 結果の集計
    jsi_values = [x for x in metrics_log["JSI"] if not np.isnan(x)]
    iou_values = [x for x in metrics_log["IoU"] if not np.isnan(x)]

    mean_jsi = np.mean(jsi_values) if jsi_values else 0.0
    mean_iou = np.mean(iou_values) if iou_values else 0.0
    
    # CSVに平均値を書き込み
    csv_writer.writerow([]) # 空行
    csv_writer.writerow(['Average', mean_jsi, mean_iou])
    csv_file.close() # ファイルを閉じる
    
    print("-" * 30)
    print(f"Test Results:")
    print(f"Mean JSI (Jaccard): {mean_jsi:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"Metrics CSV saved to: {csv_path}")
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Melon Net Segmentation Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file used for training')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='./inference_results', help='Directory to save result images')
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み
    config = load_config(args.config)
    
    # 推論実行
    run_inference(config, args.weights, args.output)
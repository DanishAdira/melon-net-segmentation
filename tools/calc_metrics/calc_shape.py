import torch
import sys
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageFile
import re
from datetime import timedelta, datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_time_string(time_str: str):
    """'1h', '30m' などの文字列をtimedeltaに変換"""
    match = re.match(r'(\d+)([hmd])', time_str)
    if not match:
        raise ValueError("Invalid time string")
    
    value, unit = match.groups()
    value = int(value)
    
    if unit == "h":
        return timedelta(hours=value)
    elif unit == "m":
        return timedelta(minutes=value)
    elif unit == "d":
        return timedelta(days=value)
    else:
        raise ValueError("Invalid time string")

def parse_timestamp_from_filename(filename: str):
    """ファイル名から日時オブジェクトを抽出する（末尾から順に検索）"""
    stem = Path(filename).stem
    parts = stem.split("_")
    for i in range(len(parts) - 1, -1, -1):
        for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
            try:
                return datetime.strptime(parts[i], fmt)
            except ValueError:
                continue
    return None

def _extract_time_str(filename: str):
    """ファイル名から日時文字列部分のみを返す（末尾から順に検索）"""
    stem = Path(filename).stem
    parts = stem.split("_")
    for i in range(len(parts) - 1, -1, -1):
        for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
            try:
                datetime.strptime(parts[i], fmt)
                return parts[i]
            except ValueError:
                continue
    return stem

def select_images(src_dir: str, period: str, max_lookback_minutes: int, base_time: str):
    """
    指定したフォルダ内の画像をスキャンし、指定頻度（period）で目標時刻に最も近い画像を選択する。
    """
    print("Scanning image directory...")
    image_files = sorted(list(Path(src_dir).glob("*.jpg")))
    
    if not image_files:
        raise ValueError("No image files found")

    data = []
    for p in tqdm(image_files, desc="Parsing timestamps"):
        dt = parse_timestamp_from_filename(p.name)
        if dt:
            data.append({"path": str(p), "datetime": dt})
    
    if not data:
        raise ValueError("Could not parse timestamps from filenames.")

    df = pd.DataFrame(data)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 期間の設定
    start_dt_limit = df["datetime"].iloc[0]
    end_dt_limit = df["datetime"].iloc[-1]
    
    # 基準時刻の設定 (最初の日の base_time)
    base_h, base_m = map(int, base_time.split(":"))
    current_target = start_dt_limit.replace(hour=base_h, minute=base_m, second=0)

    if current_target < start_dt_limit - timedelta(days=1):
        current_target = current_target.replace(year=start_dt_limit.year, month=start_dt_limit.month, day=start_dt_limit.day)

    period_delta = parse_time_string(period)
    selected_images = []
    
    print(f"Selecting images from {start_dt_limit} to {end_dt_limit} every {period}...")
    
    while current_target <= end_dt_limit:

        search_start = current_target - timedelta(minutes=max_lookback_minutes)
        search_end = current_target 

        # 範囲内の画像を抽出
        candidates = df[(df["datetime"] >= search_start) & (df["datetime"] <= search_end)]
        
        if not candidates.empty:
            # 目標時刻との差が最も小さいものを選ぶ
            candidates = candidates.copy()
            candidates["diff"] = (candidates["datetime"] - current_target).abs()
            best_match = candidates.loc[candidates["diff"].idxmin()]
            
            if not selected_images or selected_images[-1] != best_match["path"]:
                selected_images.append(best_match["path"])

        current_target += period_delta

    print(f"Selected {len(selected_images)} images out of {len(image_files)} based on sampling period.")
    return selected_images

def filter_daytime_images(image_paths, start_hour=6, end_hour=18):
    """
    リスト内の画像パスから、指定された時間帯以外の画像を除外する。
    """
    filtered_paths = []
    print(f"Filtering images to keep only between {start_hour}:00 and {end_hour}:00...")

    for path_str in image_paths:
        dt = parse_timestamp_from_filename(Path(path_str).name)
        if dt:
            if start_hour <= dt.hour < end_hour:
                filtered_paths.append(path_str)

    print(f"Filtered down to {len(filtered_paths)} images (Daytime only).")
    return filtered_paths

def filter_by_start_date(image_paths, start_date_str):
    """
    指定された計測開始日時より前の画像を除外する。
    start_date_str: "YYYYMMDD" または "YYYYMMDD-HH" 形式の文字列
    """
    if not start_date_str:
        return image_paths

    for fmt in ("%Y%m%d-%H", "%Y%m%d"):
        try:
            start_dt = datetime.strptime(start_date_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(
            "Invalid start_date format. Please use YYYYMMDD or YYYYMMDD-HH (e.g. 20250820-06)."
        )

    filtered_paths = []
    for path_str in image_paths:
        dt = parse_timestamp_from_filename(Path(path_str).name)
        if dt and dt >= start_dt:
            filtered_paths.append(path_str)

    print(f"Filtered down to {len(filtered_paths)} images (on or after {start_date_str}).")
    return filtered_paths

def complete_mask_geometry(mask: np.ndarray, pad_size: int = 200) -> np.ndarray:
    h, w = mask.shape
    mask_padded = cv2.copyMakeBorder(mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    
    contours, _ = cv2.findContours(mask_padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return mask 
    
    contour = max(contours, key=cv2.contourArea)
    clean_points = []
    x_min, x_max = pad_size, pad_size + w - 1
    y_min, y_max = pad_size, pad_size + h - 1
    margin = 5
    
    for pt in contour:
        px, py = pt[0]
        if (x_min + margin < px < x_max - margin) and (y_min + margin < py < y_max - margin):
            clean_points.append(pt)
    
    clean_points = np.array(clean_points)
    if len(clean_points) < 10: return mask

    try:
        (xc, yc), (major, minor), angle = cv2.fitEllipse(clean_points)
        ideal_mask = np.zeros_like(mask_padded)
        cv2.ellipse(ideal_mask, ((xc, yc), (major, minor), angle), 255, -1)
        combined_mask_padded = cv2.bitwise_or(mask_padded, ideal_mask)
        final_mask = combined_mask_padded[pad_size:pad_size+h, pad_size:pad_size+w]
        return final_mask
    except Exception:
        return mask

def calculate_metrics(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0 or len(contour) < 5: return None

    # 円形度: 4πA / P^2
    circularity = (4.0 * np.pi * area) / (perimeter**2)

    # 外接矩形（幅・高さ・体積推定用）
    x, y, w, h = cv2.boundingRect(contour)

    return {
        "width_px": w,
        "height_px": h,
        "circularity": circularity,
        "estimated_volume": (4/3) * np.pi * (h/2) * ((w/2)**2)
    }

def process_selected_images(image_paths, output_csv_path_str, yolo_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_csv_path = Path(output_csv_path_str)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading YOLO model from {yolo_path}...")
    melon_model = YOLO(yolo_path).to(device)
    
    results = []
    initial_volume = None

    print(f"Processing {len(image_paths)} sampled images...")

    for img_path_str in tqdm(image_paths):
        img_path = Path(img_path_str)
        file_name = img_path.name
        
        # ファイル名から日時文字列を抽出 (末尾から順に検索)
        time_str = _extract_time_str(file_name)
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            yolo_results = melon_model(image, verbose=False)[0]
            
            if yolo_results.masks is None or len(yolo_results.masks) == 0:
                results.append({"time": time_str, "filename": file_name, "detected": False})
                continue
                
            masks = yolo_results.masks.data
            areas = masks.sum(dim=(1, 2))
            best_idx = int(torch.argmax(areas).item())
            
            raw_mask = masks[best_idx].cpu().numpy()
            if raw_mask.shape != image.shape[:2]:
                raw_mask = cv2.resize(raw_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bin = (raw_mask > 0.5).astype(np.uint8) * 255
            
            completed_mask = complete_mask_geometry(mask_bin)
            metrics = calculate_metrics(completed_mask)
            
            if metrics:
                current_volume = metrics["estimated_volume"]
                if initial_volume is None:
                    initial_volume = current_volume
                    relative_growth = 1.0
                else:
                    relative_growth = current_volume / initial_volume
                
                record = {
                    "time": time_str,
                    "filename": file_name,
                    "detected": True,
                    "estimated_volume_px3": current_volume,
                    "relative_growth": relative_growth,
                    "width_px": metrics["width_px"],
                    "height_px": metrics["height_px"],
                    "circularity": metrics["circularity"],
                }
                results.append(record)
            else:
                 results.append({"time": time_str, "filename": file_name, "detected": False, "note": "Mask too small"})

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results.append({"time": time_str, "filename": file_name, "error": str(e)})

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved sampled metrics to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="入力画像のフォルダパス")
    parser.add_argument("--output_dir", type=str, required=True, help="出力CSVファイルのフルパス (例: out.csv)")
    parser.add_argument("--yolo_path", type=str, default="/home/hidayat/MelonNetSegmentation/fruit-detect/results/runs/yolo11n-seg-melon6/weights/best.pt")
    
    # サンプリング用引数
    parser.add_argument("--period", type=str, default="1h", help="サンプリング間隔 (例: 1h, 30m)")
    parser.add_argument("--max_lookback", type=int, default=30, help="画像がない場合に遡る最大分数")
    parser.add_argument("--base_time", type=str, default="12:00", help="サンプリングの基準時刻")
    parser.add_argument("--start_date", type=str, default=None, help="計測開始日時をYYYYMMDD または YYYYMMDD-HH 形式で指定 (例: 20250820 / 20250820-06)。指定した場合、この日時より前のデータは除外されます。")

    # 時間帯フィルタ用引数 (デフォルトで6:00-18:00)
    parser.add_argument("--start_hour", type=int, default=6, help="処理対象とする開始時間 (時)")
    parser.add_argument("--end_hour", type=int, default=18, help="処理対象とする終了時間 (時, 含まない)")

    args = parser.parse_args()

    # 1. 画像の選別 (サンプリング)
    # ここで全ファイルをスキャンし、DataFrameを用いてマッチングを行います
    selected_paths = select_images(
        src_dir=args.input_dir,
        period=args.period,
        max_lookback_minutes=args.max_lookback,
        base_time=args.base_time
    )

    # 2. 時間帯によるフィルタリング (朝6時〜夜18時以外を削除)
    filtered_paths = filter_daytime_images(
        image_paths=selected_paths,
        start_hour=args.start_hour,
        end_hour=args.end_hour
    )

    if args.start_date:
        print(f"Applying start date filter (Excluding data before {args.start_date})...")
        filtered_paths = filter_by_start_date(filtered_paths, args.start_date)

    # 3. 選別された画像のみ処理
    if not filtered_paths:
        print("No images found after filtering. Exiting.")
    else:
        process_selected_images(
            image_paths=filtered_paths,
            output_csv_path_str=args.output_dir,
            yolo_path=args.yolo_path
        )
import argparse
import re
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def _find_time_part(stem: str):
    """ファイル名のステムを '_' で分割し、末尾から順に日付パターンに合う部分とそのインデックスを返す"""
    parts = stem.split("_")
    for i in range(len(parts) - 1, -1, -1):
        for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
            try:
                datetime.strptime(parts[i], fmt)
                return parts[i], i
            except ValueError:
                continue
    raise ValueError(f"Could not find a valid datetime part in filename stem: '{stem}'")

def parse_time_string(time_str: str):
    """'1h', '30m', '1d'のような時間文字列をtimedeltaオブジェクトに変換する"""
    match = re.match(r'(\d+)([hmd])', time_str)
    if not match:
        raise ValueError("Invalid time string. Use format like '1h', '30m', '1d'.")
    value, unit = match.groups()
    value = int(value)
    if unit == "h": return timedelta(hours=value)
    elif unit == "m": return timedelta(minutes=value)
    elif unit == "d": return timedelta(days=value)
    else: raise ValueError("Invalid time unit. Use 'h', 'm', or 'd'.")

def select_images(src_dir: str, period: str, max_lookback_minutes: int, base_time: str):
    """指定したフォルダから指定の頻度で画像を選択する"""
    image_files = sorted(Path(src_dir).glob("*.jpg"), key=lambda x: x.name)
    if not image_files:
        print(f"Warning: No image files found in {src_dir}")
        return []
    try:
        start_time_str, time_idx = _find_time_part(image_files[0].stem)
        end_time_str, _ = _find_time_part(image_files[-1].stem)
        prefix = "_".join(image_files[0].stem.split("_")[:time_idx]) + "_"
        for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
            try:
                start_dt = datetime.strptime(start_time_str, fmt)
                end_dt = datetime.strptime(end_time_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized datetime format: '{start_time_str}'")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse date from filenames. Error: {e}")
    
    start_dt = start_dt.replace(hour=int(base_time.split(":")[0]), minute=int(base_time.split(":")[1]), second=0)
    current_time = start_dt
    selected_images = []
    
    while current_time <= end_dt:
        image_found = False
        for i in range(max_lookback_minutes + 1):
            tmp_current_time = current_time - timedelta(minutes=i)
            potential_paths = list(Path(src_dir).glob(f"{prefix}{tmp_current_time.strftime('%Y%m%d-%H%M')}*.jpg"))
            if potential_paths:
                selected_images.append(str(potential_paths[0]))
                image_found = True
                break
        current_time += parse_time_string(period)
    return sorted(list(set(selected_images)))

def delete_night_images(image_paths, morning_time="06:00", night_time="18:00"):
    """朝6時から夜6時までの画像以外を削除"""
    selected_images = []
    morning_h, night_h = int(morning_time.split(":")[0]), int(night_time.split(":")[0])
    for image_path in image_paths:
        try:
            time_str, _ = _find_time_part(Path(image_path).stem)
            for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
                try:
                    img_time = datetime.strptime(time_str, fmt)
                    break
                except ValueError:
                    continue
            if morning_h <= img_time.hour < night_h:
                selected_images.append(image_path)
        except (ValueError, IndexError):
            pass
    return selected_images

def filter_by_start_date(image_paths, start_date_str):
    """
    指定された計測開始日より前の画像を除外する。
    start_date_str: "YYYYMMDD" 形式の文字列
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

    filtered_images = []
    for image_path in image_paths:
        try:
            time_str, _ = _find_time_part(Path(image_path).stem)
            for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
                try:
                    img_time = datetime.strptime(time_str, fmt)
                    break
                except ValueError:
                    continue
            if img_time >= start_dt:
                filtered_images.append(image_path)
        except (ValueError, IndexError):
            pass

    return filtered_images

def filter_by_harvest_date(image_paths, harvest_date_str):
    """
    指定された収穫日(harvest_date)以降の画像を除外する。
    harvest_date_str: "YYYYMMDD" 形式の文字列
    """
    if not harvest_date_str:
        return image_paths

    try:
        harvest_dt = datetime.strptime(harvest_date_str, "%Y%m%d")
    except ValueError:
        raise ValueError("Invalid harvest_date format. Please use YYYYMMDD.")

    filtered_images = []
    for image_path in image_paths:
        try:
            time_str, _ = _find_time_part(Path(image_path).stem)
            for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
                try:
                    img_time = datetime.strptime(time_str, fmt)
                    break
                except ValueError:
                    continue
            if img_time < harvest_dt:
                filtered_images.append(image_path)
        except (ValueError, IndexError):
            pass

    return filtered_images

def binarize_image(image_array, threshold=100):
    img_blur = cv2.blur(image_array, (9, 9))
    _, binary_image = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def thin_image(binary_image):
    binary_normalized = binary_image // 255
    skelton = skeletonize(binary_normalized).astype(np.uint8) * 255
    return skelton

def calculate_branch_points(skelton_image, eps=5):
    """細線化画像から分岐点数を算出"""
    if skelton_image.sum() == 0: return 0
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    filtered = convolve(skelton_image // 255, kernel, mode='constant', cval=0)
    raw_branch_points = np.column_stack(np.where(filtered >= 13))
    if len(raw_branch_points) == 0: return 0
    clustering = DBSCAN(eps=eps, min_samples=1).fit(raw_branch_points)
    return len(set(clustering.labels_))

def calculate_orientation_metrics(skelton_image):
    """
    分岐点除去によるセグメント方位解析
    細線化画像を線分に分解し、縦成分・横成分・縦横比を算出する。
    """
    if skelton_image.sum() == 0:
        return 0, 0, np.nan

    # 1. 畳み込みを用いて分岐点(交差点)を検出
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    normalized_skel = skelton_image // 255
    filtered = convolve(normalized_skel, kernel, mode='constant', cval=0)
    
    # 分岐点マスクの作成
    branch_mask = (filtered >= 13)
    
    # 2. 分岐点を除去してセグメント化（切断）
    segments_img = normalized_skel.copy()
    segments_img[branch_mask] = 0
    
    # 3. 各セグメントのラベリング
    label_img = label(segments_img, connectivity=2)
    regions = regionprops(label_img)
    
    v_comp = 0  # 縦成分の総画素数
    h_comp = 0  # 横成分の総画素数
    
    for props in regions:
        # 極端に小さいノイズは無視（3ピクセル未満など）
        if props.area < 3:
            continue
            
        # orientation は [-pi/2, pi/2] の範囲 (ラジアン)
        angle = abs(props.orientation)
        
        # 45度 (pi/4) を閾値として分類
        if angle > (math.pi / 4):
            v_comp += props.area
        else:
            h_comp += props.area
            
    # 4. 縦横比 (V/H Ratio) の算出
    if h_comp == 0:
        if v_comp > 0:
            vh_ratio = np.nan
        else:
            vh_ratio = np.nan
    else:
        vh_ratio = v_comp / h_comp
        
    return v_comp, h_comp, vh_ratio

def calculate_overall_density(binary_image):
    return np.sum(binary_image > 0) / binary_image.size

def calculate_all_metrics(image_paths, output_path, melon_id, pollination_date=None):
    """
    画像のリストから全ての指標を計算し、CSVファイルに出力する。
    """
    pollination_dt = None
    if pollination_date:
        try:
            pollination_dt = datetime.strptime(pollination_date, "%Y%m%d")
        except ValueError:
            raise ValueError("Invalid pollination_date format. Please use YYYYMMDD.")

    results = []
    columns = [
        "melon", "time", "days_after_pollination",
        "overall_density", "branch_points",
        "v_component", "h_component", "vh_ratio",
    ]

    print(f"Calculating metrics for {len(image_paths)} images...")
    for image_path in tqdm(image_paths):
        row_data = {"melon": melon_id}
        path_obj = Path(image_path)

        # --- 日付情報の解析 ---
        try:
            time_str, _ = _find_time_part(path_obj.stem)
            row_data["time"] = time_str
            image_date_str = time_str.split('-')[0]
            image_dt = datetime.strptime(image_date_str, "%Y%m%d")
            if pollination_dt:
                row_data["days_after_pollination"] = (image_dt - pollination_dt).days
            else:
                row_data["days_after_pollination"] = np.nan
        except (ValueError, IndexError):
            row_data["time"] = path_obj.stem
            row_data["days_after_pollination"] = np.nan

        # --- 画像読み込みと前処理 ---
        mask_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"Warning: Could not read {image_path}. Skipping.")
            continue
        
        binary_image_for_metrics = binarize_image(mask_image)
        skelton_image = thin_image(binary_image_for_metrics)

        # --- 指標の算出 ---
        density = calculate_overall_density(binary_image_for_metrics)
        branch_points = calculate_branch_points(skelton_image)
        v_comp, h_comp, vh_ratio = calculate_orientation_metrics(skelton_image)

        # データの格納
        row_data["overall_density"] = density
        row_data["branch_points"] = branch_points
        row_data["v_component"] = h_comp
        row_data["h_component"] = v_comp
        row_data["vh_ratio"] = h_comp / v_comp if v_comp != 0 else np.nan
            
        results.append(row_data)

    if not results:
        print("No results to save.")
        return
        
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved results to {output_path}")

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(description="メロンの網目画像の各種指標を計算し、CSVに出力します。")
    arg_parse.add_argument("--src_dir", type=str, required=True, help="マスク画像が含まれるディレクトリのパス。")
    arg_parse.add_argument("--output_path", type=str, required=True, help="出力するCSVファイルのパス。")
    arg_parse.add_argument("--melon_id", type=str, required=True, help="個体ID（メロンの識別子）")
    arg_parse.add_argument("--period", type=str, default="1h", help="画像の選択頻度 (例: '1h', '30m', '1d')。")
    arg_parse.add_argument("--max_lookback_minutes", type=int, default=10, help="指定時刻に画像がない場合、何分前まで遡って探すか。")
    arg_parse.add_argument("--base_time", type=str, default="12:00", help="各日の選択期間を開始する基準時刻 (例: '09:00')。")
    arg_parse.add_argument("--pollination_date", type=str, default=None, help="交配日をYYYYMMDD形式で指定 (例: 20250815)。")
    arg_parse.add_argument("--harvest_date", type=str, default=None, help="収穫日をYYYYMMDD形式で指定。指定した場合、この日以降のデータは除外されます。")
    arg_parse.add_argument("--start_date", type=str, default=None, help="計測開始日時をYYYYMMDD または YYYYMMDD-HH 形式で指定 (例: 20250820 / 20250820-06)。指定した場合、この日時より前のデータは除外されます。")
    arg_parse.add_argument("--disable_daylight_filter", action="store_true", help="このフラグを立てると、日中（6時〜18時）のフィルタリングを無効にします。")
    
    args = arg_parse.parse_args()

    # 画像選択
    selected_image_paths = select_images(
        src_dir=args.src_dir, period=args.period,
        max_lookback_minutes=args.max_lookback_minutes, base_time=args.base_time
    )
    
    # 日中のフィルタリング
    if not args.disable_daylight_filter:
        print("Applying daylight filter (06:00 - 18:00)...")
        selected_image_paths = delete_night_images(selected_image_paths)

    if args.start_date:
        print(f"Applying start date filter (Excluding data before {args.start_date})...")
        selected_image_paths = filter_by_start_date(selected_image_paths, args.start_date)

    # 収穫日以降のデータ除外
    if args.harvest_date:
        print(f"Applying harvest date filter (Excluding data on or after {args.harvest_date})...")
        selected_image_paths = filter_by_harvest_date(selected_image_paths, args.harvest_date)

    # 計算実行
    if selected_image_paths:
        calculate_all_metrics(
            image_paths=selected_image_paths, output_path=args.output_path,
            melon_id=args.melon_id, pollination_date=args.pollination_date
        )
    else:
        print("No images were selected based on the criteria.")
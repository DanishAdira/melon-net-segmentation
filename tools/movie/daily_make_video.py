import argparse
import cv2
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="画像フォルダから6時-18時のデータを1時間ごとに抽出し、日付と経過日数入りタイムラプス動画を作成します。")
    parser.add_argument("--input_dir", type=str, required=True, help="画像が入っている入力フォルダのパス")
    parser.add_argument("--output_path", type=str, required=True, help="出力する動画ファイルのパス (例: output.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="動画のフレームレート (デフォルト: 10)")
    parser.add_argument("--pollination_date", type=str, default=None, help="交配日 (例: 20230801)。指定すると経過日数を表示します。")
    return parser.parse_args()

def parse_datetime_from_filename(filename):
    """
    ファイル名から日時オブジェクトを生成する
    対応形式: rt_01_01_HDR_20230809-1611.jpg -> 2023年8月9日 16:11
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    time_str = parts[-1]
    
    try:
        dt = datetime.strptime(time_str, "%Y%m%d-%H%M%S")
    except ValueError:
        try:
            dt = datetime.strptime(time_str, "%Y%m%d-%H%M")
        except ValueError:
            return None
    return dt

def filter_and_select_images(input_dir):
    """
    画像を読み込み、以下の条件でフィルタリングと選択を行う
    1. 時間帯: 6:00 <= hour <= 18:00
    2. 頻度: 1時間ごとに1枚
    """
    src_path = Path(input_dir)
    image_files = sorted(list(src_path.glob("*.jpg")))
    
    if not image_files:
        print(f"エラー: 指定されたフォルダに .jpg ファイルが見つかりません: {input_dir}")
        return []

    print(f"フォルダ内の全画像数: {len(image_files)} 枚")
    
    valid_images = []
    for p in image_files:
        dt = parse_datetime_from_filename(p.name)
        if dt:
            valid_images.append({"path": str(p), "dt": dt})
    
    if not valid_images:
        print("エラー: ファイル名から日時を解析できませんでした。")
        return []

    # 1. 時間帯フィルタリング (6:00 - 18:00)
    daytime_images = []
    for img in valid_images:
        h = img["dt"].hour
        if 6 <= h <= 18:
            daytime_images.append(img)
            
    print(f"6時〜18時の画像数: {len(daytime_images)} 枚")

    # 2. 1時間ごとに選択
    selected_images = []
    seen_hours = set()

    for img in daytime_images:
        hour_key = img["dt"].strftime("%Y%m%d-%H")
        if hour_key not in seen_hours:
            selected_images.append(img["path"])
            seen_hours.add(hour_key)
    
    print(f"1時間ごとに抽出後の画像数: {len(selected_images)} 枚")
    return sorted(selected_images)

def draw_info_on_image(img, dt, pollination_dt=None):
    """
    画像の上部に日付と（あれば）経過日数を描画する
    """
    if dt is None:
        return img
    
    # 基本の日付テキスト
    text = dt.strftime("%Y/%m/%d %H:%M")
    
    # 交配日が指定されている場合、日数を計算して追加
    if pollination_dt is not None:
        # 時間の差分を計算し、日数を取り出す
        delta = dt - pollination_dt
        days = delta.days
        text += f" (Day {days})"
    
    h, w = img.shape[:2]
    
    # フォント設定（画像の幅に合わせてサイズを自動調整）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 1200.0
    if font_scale < 0.5: font_scale = 0.5
    
    thickness = int(font_scale * 2)
    if thickness < 1: thickness = 1

    # テキストサイズを取得して中央揃えの位置を計算
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_w) // 2
    
    # 上部に表示（上端から少し余白を空ける）
    # text_h は文字の高さ。これに余白(高さの5%程度)を足した位置をベースラインにする
    y = text_h + int(h * 0.05)

    # 黒い縁取り
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
    # 白い文字
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img

def create_video(image_paths, output_path, fps, pollination_dt):
    """
    画像に日付・経過日数を書き込んで動画を作成する
    """
    if not image_paths:
        print("動画にする画像が選択されませんでした。")
        return

    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"エラー: 画像を読み込めませんでした: {image_paths[0]}")
        return
        
    h, w, layers = first_img.shape
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"動画の作成を開始します: {output_path}")
    
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is not None:
            # 日時を取得して描画
            dt = parse_datetime_from_filename(Path(img_path).name)
            img = draw_info_on_image(img, dt, pollination_dt)
            
            out.write(img)
        else:
            print(f"警告: スキップします: {img_path}")

    out.release()
    print("完了しました。")

def main():
    args = parse_args()
    
    # 交配日の解析
    pollination_dt = None
    if args.pollination_date:
        try:
            pollination_dt = datetime.strptime(args.pollination_date, "%Y%m%d")
            print(f"交配日を設定しました: {pollination_dt.strftime('%Y/%m/%d')}")
        except ValueError:
            print("エラー: 交配日の形式が不正です。YYYYMMDD形式で指定してください (例: 20230801)")
            return

    selected_paths = filter_and_select_images(args.input_dir)
    create_video(selected_paths, args.output_path, args.fps, pollination_dt)

if __name__ == "__main__":
    main()
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def create_video_from_frames(input_dir, output_path, fps):
    """
    連番のフレーム画像(frame_xxxxx.jpg)を読み込み、動画ファイルを作成する
    
    Args:
        input_dir (str): フレーム画像が保存されているディレクトリパス
        output_path (str): 出力する動画ファイルのパス (.mp4)
        fps (int): 動画のフレームレート
    """
    src_path = Path(input_dir)
    
    # ファイル名順にソートして取得 (frame_00000.jpg, frame_00001.jpg, ...)
    image_paths = sorted(list(src_path.glob("frame_*.jpg")))

    if not image_paths:
        print(f"Error: No 'frame_*.jpg' images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} frames. Start creating video...")

    video_writer = None

    for img_path in tqdm(image_paths):
        # 画像読み込み
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue

        # 初回のみVideoWriterを初期化
        if video_writer is None:
            h, w, _ = img.shape
            # mp4形式で保存
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # 書き込み
        video_writer.write(img)
    
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved successfully to: {output_path}")
    else:
        print("Error: Video writer could not be initialized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sequential frame images to video.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing frame_*.jpg images")
    parser.add_argument("--output_path", type=str, required=True, help="Output video path (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    
    args = parser.parse_args()
    
    # 親ディレクトリがない場合は作成
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    create_video_from_frames(
        input_dir=args.input_dir,
        output_path=args.output_path,
        fps=args.fps
    )
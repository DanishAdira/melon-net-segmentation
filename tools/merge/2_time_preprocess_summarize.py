import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# コマンドライン引数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="生育指標CSVに時系列平滑化処理（移動中央値→単調増加補正→移動平均）を適用する。"
                     "リサンプリングは行わず、元の時間解像度のまま出力する（日次集計は別ファイルで行う）。"
    )
    parser.add_argument("--input_dir",           type=str,   required=True, help="入力CSVが格納されたフォルダパス。")
    parser.add_argument("--output_csv",           type=str,   required=True, help="出力CSVのファイルパス。")
    parser.add_argument("--rolling_window",       type=int,   default=5,     help="移動窓サイズ（デフォルト: 5）。")
    parser.add_argument("--monotonic_tolerance",  type=float, default=0.05,  help="単調増加補正の許容割合（デフォルト: 0.05）。")
    return parser.parse_args()

# =============================================================================
# 設定
# =============================================================================

NON_METRIC_COLS = {"melon", "time", "days_after_pollination", "filename", "detected"}
LEADING_COLS    = ["melon", "time", "Datetime", "days_after_pollination", "filename"]

# =============================================================================
# 時系列処理
# =============================================================================

def _apply_timeseries(series: pd.Series, rolling_window: int, monotonic_tolerance: float) -> pd.Series:
    # Step 1: 移動中央値
    y = series.rolling(window=rolling_window, center=True, min_periods=1).median()

    # Step 2: 許容範囲付き単調増加補正
    y_vals = y.to_numpy(dtype=float)
    for i in range(1, len(y_vals)):
        if y_vals[i] < y_vals[i - 1] * (1 - monotonic_tolerance):
            y_vals[i] = y_vals[i - 1]

    # Step 3: 移動平均
    y_smooth = (
        pd.Series(y_vals, index=series.index)
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean()
    )
    return y_smooth


def process_file(csv_path: Path, rolling_window: int, monotonic_tolerance: float) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path, dtype={"melon": str})
    except Exception as e:
        print(f"  [ERROR] 読み込み失敗: {csv_path.name} ({e})")
        return None

    if "melon" not in df.columns:
        print(f"  [WARN]  melon列なし: {csv_path.name} → スキップ")
        return None

    melon_ids = df["melon"].dropna().unique()
    if len(melon_ids) == 0:
        print(f"  [WARN]  melon列が空: {csv_path.name} → スキップ")
        return None
    if len(melon_ids) > 1:
        print(f"  [WARN]  melon列に複数の個体IDが混在しています: {csv_path.name} {list(melon_ids)} → 先頭の値を使用")
    melon_id = melon_ids[0]

    if "days_after_pollination" not in df.columns:
        print(f"  [WARN]  days_after_pollination なし: {csv_path.name}")
        return None

    df["days_after_pollination"] = pd.to_numeric(df["days_after_pollination"], errors="coerce")
    df = df.dropna(subset=["days_after_pollination"]).sort_values("days_after_pollination").reset_index(drop=True)

    if df.empty:
        return None

    # ── time列からDatetime列（YYYYMMDD）を生成
    # time列のフォーマット例: 20251126-1500 → 20251126
    if "time" in df.columns:
        df["Datetime"] = df["time"].astype(str).str.extract(r"^(\d{8})")[0]
    else:
        print(f"  [WARN]  time列なし: {csv_path.name} → Datetime列は空になります")
        df["Datetime"] = None

    metric_cols = [
        c for c in df.columns
        if c not in NON_METRIC_COLS | {"Datetime"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    keep_cols = ["days_after_pollination", "Datetime"]
    if "time" in df.columns:
        keep_cols.append("time")
    if "filename" in df.columns:
        keep_cols.append("filename")

    processed = df[keep_cols].copy()
    for col in metric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        processed[col] = _apply_timeseries(s, rolling_window, monotonic_tolerance).to_numpy(dtype=float)

    processed.insert(0, "melon", melon_id)
    return processed

# =============================================================================
# メイン処理
# =============================================================================

def build_smoothed_csv(input_dir: str, output_csv: str, rolling_window: int, monotonic_tolerance: float):
    csv_files = sorted(Path(input_dir).glob("*.csv"))
    if not csv_files:
        print(f"CSVファイルが見つかりません: {input_dir}")
        return

    all_frames = []

    for csv_path in csv_files:
        print(f"処理中: {csv_path.name}")
        result = process_file(csv_path, rolling_window, monotonic_tolerance)
        if result is not None:
            all_frames.append(result)
        else:
            print(f"  → スキップ")

    if not all_frames:
        print("有効な結果がありませんでした。")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # 列順: melon, time, Datetime, days_after_pollination, その他
    leading_present = [c for c in LEADING_COLS if c in combined.columns]
    other_cols = [c for c in combined.columns if c not in LEADING_COLS]
    combined = combined[leading_present + other_cols]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\n保存完了: {output_csv}")
    print(f"行数: {len(combined)}, 個体数: {combined['melon'].nunique()}")
    return combined


if __name__ == "__main__":
    args = parse_args()
    build_smoothed_csv(
        input_dir           = args.input_dir,
        output_csv          = args.output_csv,
        rolling_window      = args.rolling_window,
        monotonic_tolerance = args.monotonic_tolerance,
    )
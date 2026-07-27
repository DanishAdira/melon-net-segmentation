import argparse
import pandas as pd
from pathlib import Path

# =============================================================================
# コマンドライン引数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="3_merge_sensor.py の出力（時刻粒度のまま平滑化・センサー結合済み）を読み込み、"
                     "個体(melon)ごとに days_after_pollination 単位で日次リサンプリング（数値列は平均集約）する。"
    )
    parser.add_argument("--input_csv",  type=str, required=True, help="入力CSVファイルパス（3_merge_sensor.pyの出力）。")
    parser.add_argument("--output_csv", type=str, required=True, help="出力CSVファイルパス。")
    return parser.parse_args()

# =============================================================================
# 設定
# =============================================================================

# 集約対象から除外する列（グルーピングキー、または日次では意味を持たない列）
NON_METRIC_COLS = {"melon", "time", "Datetime", "days_after_pollination", "filename", "detected"}
LEADING_COLS    = ["melon", "Datetime", "days_after_pollination"]

# =============================================================================
# メイン処理
# =============================================================================

def resample_daily(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv, dtype={"melon": str})

    if "melon" not in df.columns:
        print(f"[ERROR] melon列が見つかりません: {input_csv}")
        exit(1)
    if "days_after_pollination" not in df.columns:
        print(f"[ERROR] days_after_pollination列が見つかりません: {input_csv}")
        exit(1)

    print(f"読み込み完了: {len(df)}行  個体数: {df['melon'].nunique()}")

    df["days_after_pollination"] = pd.to_numeric(df["days_after_pollination"], errors="coerce")
    df = df.dropna(subset=["days_after_pollination"])
    df["days_after_pollination"] = df["days_after_pollination"].round().astype(int)

    # time列（時分粒度）は日次では単一の値に定まらないため集約対象から除外し、
    # Datetime列（日付のみ）で代表させる。
    metric_cols = [
        c for c in df.columns
        if c not in NON_METRIC_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    agg_dict = {col: "mean" for col in metric_cols}
    if "Datetime" in df.columns:
        agg_dict["Datetime"] = "first"

    daily = (
        df.groupby(["melon", "days_after_pollination"], as_index=False, sort=True)
        .agg(agg_dict)
    )

    # 列順: melon, Datetime, days_after_pollination, その他
    leading_present = [c for c in LEADING_COLS if c in daily.columns]
    other_cols = [c for c in daily.columns if c not in LEADING_COLS]
    daily = daily[leading_present + other_cols]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False)
    print(f"\n保存完了: {output_csv}")
    print(f"行数: {len(daily)}, 個体数: {daily['melon'].nunique()}")
    return daily


if __name__ == "__main__":
    args = parse_args()
    resample_daily(input_csv=args.input_csv, output_csv=args.output_csv)

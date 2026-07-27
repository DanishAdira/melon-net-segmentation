import argparse
from zoneinfo import ZoneInfo

import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge melon metrics CSV and sensor CSV on nearest timestamp."
    )
    parser.add_argument("--metrics", required=True, help="メロンの網目・形状の生育指標CSV (has 'time' column)")
    parser.add_argument("--sensor", required=True, help="生のセンサーデータCSV (has 'timestamp' column)")
    parser.add_argument("-o", "--output", default="merged_sensor.csv", help="Output CSV path")
    parser.add_argument(
        "--tolerance",
        default="1h",
        help="Max time gap allowed for nearest match (e.g. '1h', '30min'). Default: 1h",
    )
    parser.add_argument(
        "--direction",
        default="nearest",
        choices=["backward", "forward", "nearest"],
        help="merge_asof direction (default: nearest)",
    )
    return parser.parse_args()


def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"melon": str})
    df["_dt"] = pd.to_datetime(df["time"], format="%Y%m%d-%H%M").dt.tz_localize(JST)
    return df.sort_values("_dt").reset_index(drop=True)


def load_sensor(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_dt"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(JST)
    return df.sort_values("_dt").reset_index(drop=True)


def main():
    args = parse_args()

    metrics = load_metrics(args.metrics)
    sensor = load_sensor(args.sensor)

    tolerance = pd.Timedelta(args.tolerance)

    merged = pd.merge_asof(
        metrics,
        sensor,
        on="_dt",
        tolerance=tolerance,
        direction=args.direction,
    )

    merged = merged.drop(columns=["_dt"])
    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()

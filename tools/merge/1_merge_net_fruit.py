import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge two CSV files on the 'time' column.")
    parser.add_argument("--csv1", help="First CSV file (e.g. net metrics)")
    parser.add_argument("--csv2", help="Second CSV file (e.g. shape metrics)")
    parser.add_argument("-o", "--output", default="merged.csv", help="Output CSV file path (default: merged.csv)")
    parser.add_argument(
        "--how",
        default="outer",
        choices=["inner", "outer", "left", "right"],
        help="Join type (default: outer)",
    )
    args = parser.parse_args()

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    merged = pd.merge(df1, df2, on="time", how=args.how)
    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()

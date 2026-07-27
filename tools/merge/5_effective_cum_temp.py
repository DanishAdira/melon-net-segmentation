import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# コマンドライン引数
# =============================================================================

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(
        description="マージ済みCSVを読み込み、有効積算温度・積算照度（計測開始起点・網目発生日起点）を算出して出力する。"
    )
    arg_parse.add_argument("--input_csv",             type=str,   required=True,  help="入力CSVファイルパス。")
    arg_parse.add_argument("--output_csv",            type=str,   required=True,  help="出力CSVファイルパス。")
    arg_parse.add_argument("--base_temperature",      type=float, default=15.0,   help="生育下限温度（デフォルト: 15.0℃）。")
    arg_parse.add_argument("--density_threshold",     type=float, default=0.1,    help="網目発生判定: overall_density の閾値（デフォルト: 0.1）。")
    arg_parse.add_argument("--vcomp_threshold",       type=float, default=250.0,  help="網目発生判定: v_component の閾値（デフォルト: 250.0）。")
    arg_parse.add_argument("--hcomp_threshold",       type=float, default=250.0,  help="網目発生判定: h_component の閾値（デフォルト: 250.0）。")
    arg_parse.add_argument("--use_hcomp",             action="store_true",         help="横ネット(h_component)を網目発生判定条件に含める場合に指定。")
    args = arg_parse.parse_args()

    # =============================================================================
    # 読み込み
    # =============================================================================

    df = pd.read_csv(args.input_csv)
    df["days_after_pollination"] = pd.to_numeric(df["days_after_pollination"], errors="coerce")
    df["temperature"]            = pd.to_numeric(df["temperature"],            errors="coerce")
    df["i_v_light"]              = pd.to_numeric(df.get("i_v_light",  np.nan), errors="coerce")
    df["Datetime"]               = df["Datetime"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    print(f"読み込み完了: {len(df)}行  個体数: {df['melon'].nunique()}")
    print(f"横ネット条件: {'有効' if args.use_hcomp else '無効'}")

    # =============================================================================
    # 網目発生日の算出
    # =============================================================================

    def mesh_onset_day(
        group: pd.DataFrame,
        density_thr: float,
        vcomp_thr: float,
        hcomp_thr: float,
        use_hcomp: bool,
    ) -> tuple[int | None, str]:
        """
        以下のいずれかを最初に満たした days_after_pollination を返す。
          - overall_density > density_thr
          - v_component     > vcomp_thr
          - h_component     > hcomp_thr  （use_hcomp=True のときのみ）

        Returns
        -------
        (onset_day, trigger_reason)
        """
        candidates = []  # (day, reason)

        if "overall_density" in group.columns:
            sub = group.dropna(subset=["overall_density"])
            hit = sub.loc[sub["overall_density"] > density_thr, "days_after_pollination"]
            if not hit.empty:
                candidates.append((hit.iloc[0], f"density>{density_thr}"))

        if "v_component" in group.columns:
            sub = group.dropna(subset=["v_component"])
            hit = sub.loc[sub["v_component"] > vcomp_thr, "days_after_pollination"]
            if not hit.empty:
                candidates.append((hit.iloc[0], f"v_comp>{vcomp_thr}"))

        if use_hcomp and "h_component" in group.columns:
            sub = group.dropna(subset=["h_component"])
            hit = sub.loc[sub["h_component"] > hcomp_thr, "days_after_pollination"]
            if not hit.empty:
                candidates.append((hit.iloc[0], f"h_comp>{hcomp_thr}"))

        if not candidates:
            return None, "未検出"

        best_day, reason = min(candidates, key=lambda x: x[0])
        return int(best_day), reason

    # =============================================================================
    # 起点別の累積列を生成するヘルパー
    # =============================================================================

    def cumulate_from_posting(
        series: pd.Series,
        days: pd.Series,
        posting_day: int,
    ) -> pd.Series:
        """
        posting_day 当日を 0 として、翌日以降の cumsum を返す。
        posting_day より前は NaN。
        """
        result = pd.Series(np.nan, index=series.index)
        posting_idx = days.index[days == posting_day]
        if not posting_idx.empty:
            result[posting_idx] = 0.0
        mask_after = days > posting_day
        if mask_after.any():
            result[mask_after] = series[mask_after].cumsum().values
        return result.round(4)

    def relative_growth_from_posting(
        volume: pd.Series,
        days: pd.Series,
        posting_day: int,
    ) -> pd.Series:
        """
        網目発生日の estimated_volume_px3 を基準(1.0)とした相対肥大度を返す。
        posting_day より前は NaN。
        posting_day 当日は 1.0。
        """
        result = pd.Series(np.nan, index=volume.index)
        posting_idx = days.index[days == posting_day]
        if posting_idx.empty:
            return result

        base_vol = volume.loc[posting_idx].iloc[0]
        if pd.isna(base_vol) or base_vol == 0:
            return result

        mask = days >= posting_day
        result[mask] = (volume[mask] / base_vol).round(6)
        return result

    # =============================================================================
    # 個体ごとに積算値を算出
    # =============================================================================

    all_frames = []

    for melon_id, group in df.groupby("melon", sort=True):
        group = group.sort_values("days_after_pollination").reset_index(drop=True)
        days  = group["days_after_pollination"]

        # ── 有効積温の日次寄与
        eff_temp  = (group["temperature"] - args.base_temperature).clip(lower=0)
        # ── 照度の日次値（負値は0に）
        eff_light = group["i_v_light"].clip(lower=0)

        # ── 1. 計測開始起点の累積
        group["effective_cumulative_temperature"] = eff_temp.cumsum().round(4)
        group["cumulative_light"]                 = eff_light.cumsum().round(4)

        # ── 2. 網目発生日の算出
        posting_day, reason = mesh_onset_day(
            group,
            args.density_threshold,
            args.vcomp_threshold,
            args.hcomp_threshold,
            args.use_hcomp,
        )

        if posting_day is not None:
            print(f"  melon={melon_id}  網目発生日: {posting_day}日目  (トリガー: {reason})")

            group["effective_cumulative_temperature_from_posting"] = cumulate_from_posting(
                eff_temp, days, posting_day
            )
            group["cumulative_light_from_posting"] = cumulate_from_posting(
                eff_light, days, posting_day
            )
            if "estimated_volume_px3" in group.columns:
                vol = pd.to_numeric(group["estimated_volume_px3"], errors="coerce")
                group["relative_growth_from_posting"] = relative_growth_from_posting(
                    vol, days, posting_day
                )
            else:
                print(f"  [WARN] estimated_volume_px3 列なし → relative_growth_from_posting は NaN")
                group["relative_growth_from_posting"] = np.nan

        else:
            print(f"  melon={melon_id}  網目発生日: 算出不可 → from_posting列はすべてNaN")
            group["effective_cumulative_temperature_from_posting"] = np.nan
            group["cumulative_light_from_posting"]                 = np.nan
            group["relative_growth_from_posting"]                  = np.nan

        all_frames.append(group)

    # =============================================================================
    # 結合・出力
    # =============================================================================

    result = pd.concat(all_frames, ignore_index=True)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"\n保存完了: {args.output_csv}")
    print(f"行数: {len(result)}  個体数: {result['melon'].nunique()}")
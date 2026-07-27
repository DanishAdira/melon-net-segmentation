import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

# =============================================================================
# 設定
# =============================================================================

X_AXIS_COLUMNS = {
    "days":             "days_after_pollination",
    "ect":              "effective_cumulative_temperature",
    "ect_from_posting": "effective_cumulative_temperature_from_posting",
}
X_AXIS_LABELS = {
    "days":             "交配後日数",
    "ect":              "有効積算温度（℃・日、計測開始起点）",
    "ect_from_posting": "有効積算温度（℃・日、網目発生日起点）",
}

DEFAULT_METRICS = ["relative_growth", "overall_density,branch_points", "v_component+h_component"]

_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:cyan", "tab:pink"]

# =============================================================================
# コマンドライン引数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="生育指標CSV（3_merge_sensor.py 出力の時刻粒度データ、または "
                     "4_resample_daily.py / 5_effective_cum_temp.py 出力の日次データ）と、"
                     "元画像・マスク画像フォルダから可視化フレームを生成する。"
                     "CSVにfilename列があれば元画像をそれで直接特定し（3の出力を想定）、"
                     "なければDatetime(日付)を手がかりにフォルダを検索する（4/5の出力を想定）。"
    )
    parser.add_argument("--csv_path",  type=str, required=True, help="生育指標CSVのパス")
    parser.add_argument("--img_dir",   type=str, required=True, help="元画像（RGB）のディレクトリ")
    parser.add_argument("--mask_dir",  type=str, required=True, help="マスク画像のディレクトリ")
    parser.add_argument("--output_dir", type=str, required=True, help="出力フレームの保存先ディレクトリ")
    parser.add_argument("--melon_id",  type=str, default=None,
                         help="対象個体ID。CSVに複数個体が含まれる場合は指定必須。")
    parser.add_argument("--src_prefix", type=str, default="rt_",
                         help="filename列の元画像プレフィックス（マスクファイル名への変換に使用）。既定: rt_")
    parser.add_argument("--mask_prefix", type=str, default="pr_",
                         help="マスク画像ファイル名のプレフィックス（--src_prefixから置換）。既定: pr_")
    parser.add_argument("--base_time", type=str, default="12:00",
                         help="filenameが無い場合、または対応ファイルが見つからない場合のフォールバック検索で、"
                              "1日の中から代表画像を選ぶ基準時刻 (例: 12:00)。既定: 12:00")
    parser.add_argument(
        "--x_axis", type=str, default="days", choices=list(X_AXIS_COLUMNS.keys()),
        help="X軸に使う時間軸。days=交配後日数 / ect=有効積算温度(計測開始起点) / "
             "ect_from_posting=有効積算温度(網目発生日起点)。既定: days"
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=None,
        help=(
            "グラフの各段に表示する指標を、段の数だけ指定する（1引数=1段、段数は可変）。"
            "各引数の書式: "
            "'col' で1指標のみ表示。 "
            "'colA,colB' で2指標をスケール別（左右に別々の目盛りを持つtwin軸）として重ねる。 "
            "'colA+colB' で2指標をスケール共通（1つの軸・1組の目盛りのみ）として重ねる"
            "（例: v_component と h_component はスケールが同じなので colA+colB を使う）。"
            f" 未指定時のデフォルト: {DEFAULT_METRICS}"
        )
    )
    return parser.parse_args()

# =============================================================================
# 指標指定のパース
# =============================================================================

def parse_metric_spec(spec: str):
    """
    'col' / 'colA,colB' / 'colA+colB' をパースし (mode, [col, ...]) を返す。
    mode: 'single'（1指標） | 'twin'（左右別軸） | 'shared'（単一軸に重ね書き）
    """
    if "+" in spec:
        cols = [c.strip() for c in spec.split("+") if c.strip()]
        return "shared", cols
    if "," in spec:
        cols = [c.strip() for c in spec.split(",") if c.strip()]
        if len(cols) > 2:
            print(f"[WARN] twin軸指定は最大2指標までです。先頭2つのみ使用します: {spec}")
            cols = cols[:2]
        return "twin", cols
    return "single", [spec.strip()]

# =============================================================================
# 日付から画像を探すヘルパー
# =============================================================================

def _extract_time_of_day(path: Path):
    """ファイル名から時刻(hour, minute)を抽出する。見つからなければNoneを返す。"""
    parts = path.stem.split("_")
    for part in reversed(parts):
        for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
            try:
                dt = datetime.strptime(part, fmt)
                return dt.hour, dt.minute
            except ValueError:
                continue
    return None


def find_image_for_date(dir_path: Path, date_str: str, base_time: str):
    """
    dir_path内から date_str(YYYYMMDD) を含むファイルを探す。
    複数見つかった場合は base_time に最も近い時刻のものを1枚選ぶ。
    見つからなければ None。
    """
    candidates = sorted(dir_path.glob(f"*{date_str}*"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    base_h, base_m = map(int, base_time.split(":"))
    target_minutes = base_h * 60 + base_m

    def diff(p: Path):
        hm = _extract_time_of_day(p)
        if hm is None:
            return float("inf")
        return abs((hm[0] * 60 + hm[1]) - target_minutes)

    return min(candidates, key=diff)


def resolve_image_path(img_dir: Path, filename: str, date_str: str, base_time: str):
    """
    filename列が使える場合はそれを最優先で使用し、
    無い場合・そのファイルが存在しない場合は日付ベースのフォールバック検索を行う。
    """
    if filename:
        candidate = img_dir / filename
        if candidate.exists():
            return candidate
    return find_image_for_date(img_dir, date_str, base_time)


def resolve_mask_path(mask_dir: Path, filename: str, src_prefix: str, mask_prefix: str,
                       date_str: str, base_time: str):
    """
    filename列が使える場合は src_prefix→mask_prefix 置換でマスクファイル名を推測して使用し、
    無い場合・そのファイルが存在しない場合は日付ベースのフォールバック検索を行う。
    """
    if filename:
        mask_filename = filename.replace(src_prefix, mask_prefix, 1) if src_prefix in filename else filename
        candidate = mask_dir / mask_filename
        if candidate.exists():
            return candidate
    return find_image_for_date(mask_dir, date_str, base_time)

# =============================================================================
# 画像リサイズ
# =============================================================================

def resize_with_pad(image, target_height, target_width):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    nw, nh = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    if len(image_resized.shape) == 2:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - nh) // 2
    x_offset = (target_width - nw) // 2
    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = image_resized
    return canvas

# =============================================================================
# グラフ生成
# =============================================================================

def create_plot_image(df, current_idx, target_height, target_width, x_col, x_label, metric_specs):
    dpi = 100
    num_plots = len(metric_specs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(target_width / dpi, target_height / dpi),
                              dpi=dpi, sharex=True)
    if num_plots == 1:
        axes = [axes]

    x = df[x_col]
    current_x = x.iloc[current_idx]
    color_cycle = iter(_PALETTE * 10)

    for ax, (mode, cols) in zip(axes, metric_specs):
        if mode == "single":
            col = cols[0]
            if col not in df.columns:
                ax.set_ylabel(f"{col}（列なし）", fontsize=9)
                continue
            color = next(color_cycle)
            ax.plot(x, df[col], color="gray", alpha=0.3, linestyle="--")
            ax.plot(x.iloc[:current_idx + 1], df[col].iloc[:current_idx + 1],
                     color=color, label=col, linewidth=2)
            if pd.notna(current_x) and pd.notna(df.iloc[current_idx][col]):
                ax.scatter(current_x, df.iloc[current_idx][col], color=color, s=80, zorder=5)
            ax.set_ylabel(col, fontsize=10)
            ax.legend(loc="upper left", fontsize=8)

        elif mode == "shared":
            # スケールが共通の指標を1つの軸にまとめて重ね書きする
            plotted_any = False
            for col in cols:
                if col not in df.columns:
                    print(f"[WARN] 列が見つかりません（スキップ）: {col}")
                    continue
                plotted_any = True
                color = next(color_cycle)
                ax.plot(x, df[col], color="gray", alpha=0.15, linestyle="--")
                ax.plot(x.iloc[:current_idx + 1], df[col].iloc[:current_idx + 1],
                         color=color, label=col, linewidth=2)
                if pd.notna(current_x) and pd.notna(df.iloc[current_idx][col]):
                    ax.scatter(current_x, df.iloc[current_idx][col], color=color, s=80, zorder=5)
            if plotted_any:
                ax.set_ylabel(" / ".join(cols), fontsize=10)
                ax.legend(loc="upper left", fontsize=8)

        elif mode == "twin":
            # スケールが異なる指標を左右別軸に設定
            col_left = cols[0]
            col_right = cols[1] if len(cols) > 1 else None

            lines, labels = [], []
            if col_left in df.columns:
                color_left = next(color_cycle)
                ax.plot(x, df[col_left], color="gray", alpha=0.3, linestyle="--")
                ax.plot(x.iloc[:current_idx + 1], df[col_left].iloc[:current_idx + 1],
                         color=color_left, label=col_left, linewidth=2)
                if pd.notna(current_x) and pd.notna(df.iloc[current_idx][col_left]):
                    ax.scatter(current_x, df.iloc[current_idx][col_left], color=color_left, s=80, zorder=5)
                ax.set_ylabel(col_left, color=color_left, fontsize=10)
                ax.tick_params(axis="y", labelcolor=color_left, labelsize=9)
                lines, labels = ax.get_legend_handles_labels()
            else:
                print(f"[WARN] 列が見つかりません（スキップ）: {col_left}")

            if col_right:
                if col_right in df.columns:
                    ax_right = ax.twinx()
                    color_right = next(color_cycle)
                    ax_right.plot(x, df[col_right], color="gray", alpha=0.3, linestyle="--")
                    ax_right.plot(x.iloc[:current_idx + 1], df[col_right].iloc[:current_idx + 1],
                                   color=color_right, label=col_right, linewidth=2, linestyle="--")
                    if pd.notna(current_x) and pd.notna(df.iloc[current_idx][col_right]):
                        ax_right.scatter(current_x, df.iloc[current_idx][col_right],
                                          color=color_right, s=80, marker="d", zorder=5)
                    ax_right.set_ylabel(col_right, color=color_right, fontsize=10)
                    ax_right.tick_params(axis="y", labelcolor=color_right, labelsize=9)
                    lines_r, labels_r = ax_right.get_legend_handles_labels()
                    lines += lines_r
                    labels += labels_r
                else:
                    print(f"[WARN] 列が見つかりません（スキップ）: {col_right}")

            if lines:
                ax.legend(lines, labels, loc="upper left", fontsize=8)

        ax.grid(True, alpha=0.5)

    current_time_str = str(df.iloc[current_idx].get("time", "")).strip()
    title_text = None
    if current_time_str and current_time_str.lower() != "nan":
        for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M"):
            try:
                title_text = datetime.strptime(current_time_str, fmt).strftime("%Y-%m-%d %H:%M")
                break
            except ValueError:
                continue
    if title_text is None:
        current_date_str = str(df.iloc[current_idx].get("Datetime", ""))
        try:
            title_text = datetime.strptime(current_date_str, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            title_text = current_date_str
    axes[0].set_title(f"日時: {title_text}", fontsize=14, fontweight="bold", pad=10)
    axes[-1].set_xlabel(x_label, fontsize=10)

    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    img_plot = np.array(canvas.buffer_rgba())
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_plot

# =============================================================================
# メイン処理
# =============================================================================

def generate_frames(csv_path, img_dir, mask_dir, output_dir, melon_id, src_prefix, mask_prefix,
                     base_time, x_axis, metrics):
    TOTAL_W, TOTAL_H = 1920, 1080
    HALF_W = TOTAL_W // 2
    HALF_H = TOTAL_H // 2

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    print(f"CSVを読み込み中: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"melon": str, "Datetime": str})

    if "melon" not in df.columns:
        print("[ERROR] melon列が見つかりません。")
        return
    if "days_after_pollination" not in df.columns:
        print("[ERROR] days_after_pollination列が見つかりません。")
        return
    if "Datetime" not in df.columns:
        print("[ERROR] Datetime列が見つかりません。")
        return

    if melon_id is not None:
        df = df[df["melon"] == str(melon_id)].copy()
    else:
        unique_melons = df["melon"].dropna().unique()
        if len(unique_melons) > 1:
            print(f"[ERROR] CSVに複数個体が含まれています: {list(unique_melons)}。--melon_id で対象を指定してください。")
            return

    if df.empty:
        print("[ERROR] 対象データが空です。")
        return

    x_col = X_AXIS_COLUMNS[x_axis]
    x_label = X_AXIS_LABELS[x_axis]
    if x_col not in df.columns:
        print(f"[ERROR] X軸に指定された列 '{x_col}' がCSVに存在しません。")
        return

    df = df.dropna(subset=[x_col]).sort_values(x_col).reset_index(drop=True)
    if df.empty:
        print(f"[ERROR] '{x_col}' が有効な行がありません。")
        return

    metric_specs = [parse_metric_spec(spec) for spec in (metrics or DEFAULT_METRICS)]
    print(f"グラフ段数: {len(metric_specs)}  内訳: {list(zip((m[0] for m in metric_specs), (m[1] for m in metric_specs)))}")
    print(f"X軸: {x_col} ({x_label})")

    has_filename = "filename" in df.columns
    print(f"画像特定方法: {'filename列を優先使用（日付ベースにフォールバック）' if has_filename else '日付ベースのみ（filename列なし）'}")

    frame_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        date_str = str(row["Datetime"]).strip()
        filename = str(row["filename"]).strip() if has_filename and pd.notna(row.get("filename")) else ""

        img_path = resolve_image_path(img_dir, filename, date_str, base_time)
        mask_path = resolve_mask_path(mask_dir, filename, src_prefix, mask_prefix, date_str, base_time)

        if img_path is None or mask_path is None:
            continue

        img_melon = cv2.imread(str(img_path))
        img_mask = cv2.imread(str(mask_path))
        if img_melon is None or img_mask is None:
            continue

        img_top_left = resize_with_pad(img_melon, HALF_H, HALF_W)
        img_bottom_left = resize_with_pad(img_mask, HALF_H, HALF_W)

        try:
            img_right = create_plot_image(df, idx, TOTAL_H, HALF_W, x_col, x_label, metric_specs)
            if img_right.shape[:2] != (TOTAL_H, HALF_W):
                img_right = cv2.resize(img_right, (HALF_W, TOTAL_H))
        except Exception as e:
            print(f"グラフ生成エラー (フレーム {frame_count}): {e}")
            continue

        img_left = np.vstack((img_top_left, img_bottom_left))
        frame = np.hstack((img_left, img_right))

        save_name = out_path / f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(str(save_name), frame)
        frame_count += 1

    print(f"フレームを保存しました: {output_dir} ({frame_count} フレーム)")


if __name__ == "__main__":
    args = parse_args()
    generate_frames(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        melon_id=args.melon_id,
        src_prefix=args.src_prefix,
        mask_prefix=args.mask_prefix,
        base_time=args.base_time,
        x_axis=args.x_axis,
        metrics=args.metrics,
    )

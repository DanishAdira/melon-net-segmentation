# melon-net-segmentation

温室メロンの生育画像から，深層学習と画像処理を用いて外観特徴を定量化分析を行うためのリポジトリです．

---

## リポジトリ構成

```
melon-net-segmentation/
├── fruit-engine/
│   ├── dataset.py
│   ├── dataset_config.yaml
│   ├── train.py
│   └── train_config.yaml
│
├── net-engine/
│   ├── datasets.py
│   ├── augmentations.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   ├── unetplusplus.py
│   │   ├── deeplabv3plus.py
│   │   └── segformer.py
│   ├── loss.py
│   ├── metrics.py
│   ├── optimizer.py
│   ├── log.py
│   ├── train.py
│   └── utils.py
│
├── tools/
│   ├── segmentation/
│   │   └── make_mask.py
│   ├── calc_metrics/
│   │   ├── calc_net_metrics.py
│   │   └── calc_shape.py
│   ├── merge/
│   │   ├── 1_merge_net_fruit.py
│   │   ├── 2_time_preprocess_summarize.py
│   │   ├── 3_merge_sensor.py
│   │   ├── 4_resample_daily.py
│   │   └── 5_effective_cum_temp.py
│   └── movie/
│       ├── make_frame.py
│       ├── frames_to_video.py
│       └── daily_make_video.py
│
├── experiments/
│   ├── tuning_config.yaml
│   └── config.yaml
│
└── requirements.txt
```

## 前提条件

minelab-datasetにマウントしてください

---

## 実行手順

学習済みモデルの構築からデータ集計・可視化までの一連の処理は，以下の順序で実行します．
（TODO: 各パスはたたき台のため，実際のデータ配置に合わせて書き換えてください）

### 果実検出モデルのトレーニング（fruit-engine）

- YOLO11n-segを用いて，画像内のメロン果実領域を検出・セグメンテーションするモデルを学習します．

マスク画像からYOLOセグメンテーション用データセットを構築
```bash
python fruit-engine/dataset.py --config fruit-engine/dataset_config.yaml
```

YOLO11n-segモデルの学習
```bash
python fruit-engine/train.py --config fruit-engine/train_config.yaml
```

学習済みの重みは `train_config.yaml` の `log.project` / `log.name` で指定したディレクトリ
（既定: `fruit-engine/results/runs/yolo11n-seg-melon/weights/best.pt`）に保存されます．

### 網目検出モデルのトレーニング（net-engine）

- U-Net等のセグメンテーションモデルを用いて，網目構造を検出するモデルを学習します．

- モデル種類・データ拡張・損失関数などはexperiments/config.yamlで設定してください．

モデルの訓練の実行
```bash
python net-engine/train.py --config experiments/config.yaml
```

（任意）Optunaによるハイパーパラメータ探索

```bash
python net-engine/search_parameters.py \
    --base_config experiments/config.yaml \
    --tuning_config experiments/tuning_config.yaml \
    --n_trials 100
```

（任意）学習済みモデルの推論・評価
```bash
python net-engine/inference.py \
    --config experiments/config.yaml \
    --weights experiments/results/checkpoints/<学習時のログ名>/best_model.pth \
    --output ./inference_results
```

学習済みモデルは `experiments/config.yaml` の `log.dir_name` 以下
（`checkpoints/<タイムスタンプ_モデル名_損失関数_optimizer_lr>/best_model.pth`）に保存されます．

### 指標の算出（tools/segmentation, tools/calc_metrics）

- 学習済みのモデルを用いて，果実の形状指標・網目指標を算出します．

果実領域を検出し，学習済みnet-engineモデルで網目マスクを生成
```bash
python tools/segmentation/make_mask.py \
    --input_dir <定点カメラ画像フォルダ> \
    --output_dir <網目マスク出力先> \
    --model_path experiments/results/checkpoints/<ログ名>/best_model.pth
```

果実の形状指標（真円度・推定体積）を算出
```bash
python tools/calc_metrics/calc_shape.py \
    --input_dir <定点カメラ画像フォルダ> \
    --output_dir <shape_metrics.csvの出力パス> \
    --yolo_path fruit-engine/results/runs/yolo11n-seg-melon/weights/best.pt
```

網目指標（網目密度・分岐点数・縦/横方向の網目形成量）を算出
```bash
python tools/calc_metrics/calc_net_metrics.py \
    --src_dir <網目マスクフォルダ> \
    --output_path <net_metrics.csvの出力パス> \
    --pollination_date <交配日 YYYYMMDD>
```

### 各CSVファイルのマージ（tools/merge）

- 算出した指標CSVと環境センサーデータを分析しやすいようにマージします．

網目指標CSVと形状指標CSVを結合
```bash
python tools/merge/1_merge_net_fruit.py \
    --csv1 <net_metrics.csv> --csv2 <shape_metrics.csv> \
    -o merged.csv
```

時系列平滑化
```bash
python tools/merge/2_time_preprocess_summarize.py \
    --input_dir <統合したファイル群のフォルダ> \
    --output_csv smoothed.csv
```

センサーデータとの結合
```bash
python tools/merge/3_merge_sensor.py \
    --metrics smoothed.csv --sensor <センサーCSV> \
    -o merged_sensor.csv
```

個体ごとの日次リサンプリング
```bash
python tools/merge/4_resample_daily.py \
    --input_csv merged_sensor.csv --output_csv daily.csv
```

有効積算温度の算出
```bash
python tools/merge/5_effective_cum_temp.py \
    --input_csv daily.csv --output_csv daily_with_temp.csv
```

### 5. 動画化（tools/movie）

- 指標の推移と画像を組み合わせたタイムラプス動画を作成します．

元画像・マスク画像・指標グラフを1フレームに合成
```bash
python tools/movie/make_frame.py \
    --csv_path daily_with_temp.csv \
    --img_dir <定点カメラ画像フォルダ> \
    --mask_dir <網目マスクフォルダ> \
    --output_dir frames/
```

フレーム画像を動画に変換
```bash
python tools/movie/frames_to_video.py \
    --input_dir frames/ --output_path result.mp4
```

- 指標グラフを含めず，日付入りタイムラプス動画のみ作成する場合:

```bash
python tools/movie/daily_make_video.py \
    --input_dir <定点カメラ画像フォルダ> \
    --output_path timelapse.mp4 \
    --pollination_date <交配日 YYYYMMDD>
```

---

## 備忘録

本リポジトリは，これまでの研究成果を一度まとめたものとなっています．生長モデルへの近似や収穫時品質推定，RFEによる特徴量算出，深層状態空間モデルの実装などはリファクタリングが完了次第，随時更新予定です．

### 更新履歴

- 2026-07-27: 初版公開（データ定量化・分析パイプラインの整備）
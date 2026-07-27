import torch
from torch.utils.data import DataLoader

from datasets import MelonDataset                           # データセット定義
from augmentations import (                                 # データ拡張
    get_augmentation_train, get_augmentation_validation
)
from models import create_model                             # モデル
from loss import get_loss                                   # 損失関数
from optimizer import get_optimizer                         # 最適化関数
from metrics import iou, jaccard_index                      # 評価指標
from log import (                                           # ログ関連
    log_epoch_results, get_tensorboard_writer, log_hyperparameters, save_checkpoint, get_log_name
)
from utils import (                                         # ユーティリティ関数
    get_args, load_config, set_seed, seed_worker
)

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    
# 訓練ループ
def train(model, device, train_loader, criterion, optimizer, writer, epoch):
    model.train()
    running_loss = 0.0
    running_jsi = 0.0
    running_per_image_iou = 0.0
    running_dataset_iou = 0.0

    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        prob = torch.sigmoid(output)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        running_jsi += jaccard_index(prob, target)
        iou_score = iou(prob, target)
        running_per_image_iou += iou_score[0]
        running_dataset_iou += iou_score[1]

        writer.add_scalar("Train/Loss_per_batch", loss.item(), epoch * len(train_loader) + batch_index)

    ret_loss = {
        "DiceLoss": running_loss / len(train_loader),
    }
    ret_metrics = {
        "JSI": running_jsi / len(train_loader),
        "PerImageIoU": running_per_image_iou / len(train_loader),
        "DatasetIoU": running_dataset_iou / len(train_loader),
    }
    
    return ret_loss["DiceLoss"], ret_metrics["PerImageIoU"]


# 検証ループ
# def validate(model, device, val_loader, criterion):
#     model.eval()
#     loss = 0.0
#     jsi = 0.0
#     per_image_iou = 0.0
#     dataset_iou = 0.0

#     with torch.no_grad():
#         for data, target in val_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             prob = torch.sigmoid(output)
#             loss += criterion(prob, target).item()

#             jsi = jaccard_index(prob, target)
#             iou_score = iou(prob, target)
#             per_image_iou += iou_score[0]
#             dataset_iou += iou_score[1]

#     ret_loss = {
#         "DiceLoss": loss / len(val_loader),
#     }
#     ret_metrics = {
#         "JSI": jsi / len(val_loader),
#         "PerImageIoU": per_image_iou / len(val_loader),
#         "DatasetIoU": dataset_iou / len(val_loader),
#     }

#     return ret_loss["DiceLoss"], ret_metrics["PerImageIoU"]

# 検証ループ（修正版）
def validate(model, device, val_loader, criterion):
    model.eval()
    loss = 0.0
    jsi = 0.0
    per_image_iou = 0.0
    dataset_iou = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss += criterion(output, target).item()

            # 評価指標用にはシグモイドを適用して確率値(0~1)にする
            prob = torch.sigmoid(output)

            # 指標計算
            jsi += jaccard_index(prob, target)
            iou_score = iou(prob, target)
            per_image_iou += iou_score[0]
            dataset_iou += iou_score[1]

    ret_loss = {
        "DiceLoss": loss / len(val_loader),
    }
    ret_metrics = {
        "JSI": jsi / len(val_loader),
        "PerImageIoU": per_image_iou / len(val_loader),
        "DatasetIoU": dataset_iou / len(val_loader),
    }

    return ret_loss["DiceLoss"], ret_metrics["PerImageIoU"]

def run_training_pipeline(config):
    # シードの設定
    set_seed(config['seed'])

    # ログディレクトリとチェックポイントディレクトリの設定
    writer = get_tensorboard_writer(config)

    # デバイス設定（GPU/CPU）
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # データ変換および拡張の設定を適用
    transform_train = get_augmentation_train(config["augmentation"])
    transform_val   = get_augmentation_validation()

    # データセットの取得
    train_dataset, val_dataset = None, None
    for dir_name in config['data']['dir_name']:
        if train_dataset is None:
            train_dataset = MelonDataset(dir_name, dir_name + "train.txt", transform=transform_train)
            val_dataset = MelonDataset(dir_name, dir_name + "validation.txt", transform=transform_val)
        else:
            train_dataset += MelonDataset(dir_name, dir_name + "train.txt", transform=transform_train)
            val_dataset += MelonDataset(dir_name, dir_name + "validation.txt", transform=transform_val)

    # データローダー
    train_loader    = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'],
                                 worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config['seed']))
    val_loader      = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'],
                                 worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config['seed']))

    # モデル、損失関数、オプティマイザの定義
    model = create_model(config['model']).to(device)
    early_stopping = EarlyStopping(patience=30)
    criterion = get_loss(config['train']['loss_function'])
    optimizer = get_optimizer(model, config['train']['optimizer'])
    log_dir_name = get_log_name(config)

    best_metrics = 0.0

    for epoch in range(1, config['train']['epochs'] + 1):
        # 訓練/検証
        train_loss, train_metrics = train(model, device, train_loader, criterion, optimizer, writer, epoch)
        val_loss, val_metrics = validate(model, device, val_loader, criterion)

        # 訓練・検証データセットの結果をTensorBoardに記録
        log_epoch_results(writer, model, train_dataset, "Train", epoch, train_loss, train_metrics, device)
        log_epoch_results(writer, model, val_dataset, "Validation", epoch, val_loss, val_metrics, device)
        print(f"\rEpoch {epoch}, Train Loss: {train_loss:.6f}, Train Metrics: {train_metrics:.6f}, Val Loss: {val_loss:.6f}, Val Metrics: {val_metrics:.6f}", end='')

        # 最高のJSIを記録し、そのときのモデルを保存
        if val_metrics > best_metrics:
            best_metrics = val_metrics
            save_checkpoint(model, optimizer, epoch, log_dir_name)

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print()
            print("Early stopping")
            break
    print()
    # ハイパーパラメータとメトリクスを記録
    log_hyperparameters(writer, config, best_metrics)

    # 訓練終了後にログを閉じる
    writer.close()

    return best_metrics

if __name__ == '__main__':
    # 引数と設定ファイルの読み込み
    args = get_args()
    config_path = args.config or "/home/hidayat/MelonNetSegmentation/experiments/config.yaml"    # 変更箇所
    config = load_config(config_path)

    # 訓練パイプラインの実行
    run_training_pipeline(config)
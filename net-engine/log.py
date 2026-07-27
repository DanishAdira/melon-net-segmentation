import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# ログ名を取得する関数
def get_log_name(config):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_name = f"{timestamp}_{config['model']['name']}_{config['train']['loss_function']}_{config['train']['optimizer']['name']}_lr{config['train']['optimizer']['learning_rate']}"
    checkpoint_dir  = Path(config["log"]['dir_name']) / Path(f"checkpoints/{log_name}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

# TensorBoardのSummaryWriterを取得する関数
def get_tensorboard_writer(config):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_name = f"{timestamp}_{config['model']['name']}_{config['train']['loss_function']}_{config['train']['optimizer']['name']}_lr{config['train']['optimizer']['learning_rate']}"
    tensorboard_dir = Path(config["log"]['dir_name']) / Path(f"runs/{log_name}")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    return writer

# サンプル画像と検出結果をTensorBoardに記録する関数
def log_epoch_results(writer, model, dataset, dataset_name, epoch, loss, jsi, device):
    model.eval()

    data, target = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
    data, target = data.to(device), target.to(device)
    
    with torch.no_grad():
        output = model(data)
        prob = torch.sigmoid(output)
        detection_result = (prob > 0.5).float()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        # 入力画像
        input_data = data[i].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        axes[i, 0].imshow(input_data)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        # 正解ラベル
        axes[i, 1].imshow(target[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # 確率マップ
        axes[i, 2].imshow(prob[i].cpu().squeeze(), cmap='jet')
        axes[i, 2].set_title("Probability Map")
        axes[i, 2].axis('off')
        
        # 検出結果
        axes[i, 3].imshow(detection_result[i].cpu().squeeze(), cmap='gray')
        axes[i, 3].set_title("Detection Result")
        axes[i, 3].axis('off')

    plt.tight_layout()
    writer.add_figure(f"{dataset_name}/Prediction Result", fig, epoch)

    # エポックごとの損失とIoUをTensorBoardに記録
    writer.add_scalar(f"{dataset_name}/Loss", loss, epoch)
    writer.add_scalar(f"{dataset_name}/IoU", jsi, epoch)

    model.train()

# ハイパーパラメータとメトリクスを記録する関数
def log_hyperparameters(writer, config, best_jsi):
    hparam_dict = {
        "Dataset"               : str(config['data']['dir_name']),
        "Batch Size"            : config['data']['batch_size'],
        "Num Workers"           : config['data']['num_workers'],
        "Horizontal Flip"       : config['augmentation']['horizontal_flip']['enabled'],
        "Model"                 : config['model']['name'],
        "Loss Function"         : config['train']['loss_function'],
        "Optimizer"             : config['train']['optimizer']["name"],
        "Learning Rate"         : config['train']['optimizer']["learning_rate"],
    }

    # 各データ拡張の設定を追加
    _update_augmentation_hparams(hparam_dict, config['augmentation'])

    metric_dict = {"JSI": best_jsi}
    writer.add_hparams(hparam_dict, metric_dict)

# 各データ拡張のハイパーパラメータを追加するヘルパー関数
def _update_augmentation_hparams(hparam_dict, augmentation_cfg):
    def update_if_enabled(cfg, name):
        if cfg.get("enabled", False):
            for key, value in cfg.items():
                if key != "enabled":
                    hparam_dict.update({f"{name}_{key}": value})
                
    for name, cfg in augmentation_cfg.items():
        update_if_enabled(cfg, name)

# モデルを保存する関数
def save_checkpoint(model, optimizer, epoch, dir_name):
    path = os.path.join(dir_name, "best_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
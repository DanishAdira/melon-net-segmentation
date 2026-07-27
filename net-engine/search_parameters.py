import optuna
from train import run_training_pipeline
from utils import load_config
from datetime import datetime
import argparse

def update_parameter(trial: optuna.trial.Trial, config: dict, update_param: dict):
    """
    チューニング対象のパラメータを更新する関数
    args:
        trial: optuna.trial.Trial
        config: dict
        update_param: dict
    """
    if update_param['type'] == 'categorical':
        x = trial.suggest_categorical(
            name=update_param['name'],
            choices=update_param['choices']
        )
    elif update_param['type'] == 'discrete_uniform':
        x = trial.suggest_discrete_uniform(
            name=update_param['name'],
            los=update_param['low'],
            high=update_param['high'],
            q=update_param['q']
        )
    elif update_param['type'] == 'float':
        if 'log' in update_param:
            x = trial.suggest_float(
                name=update_param['name'],
                low=update_param['low'],
                high=update_param['high'],
                log=update_param['log']
            )
        elif 'step' in update_param:
            x = trial.suggest_float(
                name=update_param['name'],
                low=update_param['low'],
                high=update_param['high'],
                step=update_param['step']
            )
        else:
            x = trial.suggest_float(
                name=update_param['name'],
                low=update_param['low'],
                high=update_param['high']
            )
    elif update_param['type'] == 'int':
        if 'step' in update_param:
            x = trial.suggest_int(
                name=update_param['name'],
                low=update_param['low'],
                high=update_param['high'],
                step=update_param['step']
            )
        else:
            x = trial.suggest_int(
                name=update_param['name'],
                low=update_param['low'],
                high=update_param['high']
            )
    elif update_param['type'] == 'uniform':
        x = trial.suggest_uniform(
            name=update_param['name'],
            low=update_param['low'],
            high=update_param['high']
        )
    else:
        raise ValueError(f"Invalid parameter type: {update_param['type']}")
    
    config[update_param['name']] = x
    

def objective(trial: optuna.trial.Trial, base_cfg: dict, tuning_cfg: list):
    """
    Optunaの目的関数
    args:
        trial: optuna.trial.Trial
    return:
        メトリクスの値
    """
    config = base_cfg.copy()
    for update_param in tuning_cfg:
        update_parameter(trial, config, update_param)
    return run_training_pipeline(config)

def run_parameter_tuning():
    # 引数の読み込み
    parser = argparse.ArgumentParser(description="Training script for wrinkle segmentation")
    parser.add_argument('--base_config', type=str, default="/home/hidayat/MelonNetSegmentation/experiments/config.yaml", help="Path to the base config file (YAML)")
    parser.add_argument('--tuning_config', type=str, default="/home/hidayat/MelonNetSegmentation/experiments/tuning_config.yaml", help="Path to the tuning config file (YAML)")
    parser.add_argument('--output_dir', type=str, default="/home/hidayat/MelonNetSegmentation/engine/MelonNetSegmentation/experiments/results/parameter_tuning", help="Directory to save the results of parameter")
    parser.add_argument('--study_name', type=str, default="melon_segmentation_parameter", help="Name of the Optuna study")
    parser.add_argument('--n_trials', type=int, default=100, help="Number of trials for the optimization")
    args = parser.parse_args()

    # 設定の読み込み
    base_cfg = load_config(args.base_config)
    tuning_cfg = load_config(args.tuning_config)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    storage_path = f"sqlite:///{args.output_dir}/{timestamp}_{args.study_name}.db"

    # Optunaの設定
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=args.study_name, storage=storage_path, load_if_exists=True, direction="maximize")
    study.optimize(lambda trial: objective(trial, base_cfg, tuning_cfg), n_trials=args.n_trials, show_progress_bar=True)

if __name__ == "__main__":
    run_parameter_tuning()
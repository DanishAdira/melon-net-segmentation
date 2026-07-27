from torch import optim

# 最適化関数の設定を行う関数
def get_optimizer(model, params):
    optimizer_name = params['name']

    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=params['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
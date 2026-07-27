import torch
from .unet import UNet
from .deeplabv3plus import DeepLabV3Plus
from .unetplusplus import UnetPlusPlus
from .segformer import SegFormer

def create_model(params: dict):
#     """
#     モデル名に応じたモデルを作成し，パラメータを辞書として渡す関数
#     """
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # モデル名
    model_name = params["name"]

    # 共通パラメータ
    model_params = {
        "encoder_name": params["encoder_name"],
        "encoder_weights": params["encoder_weights"],
        "classes": 1,
        "activation": None
    }

    if model_name == "UNet":
        return UNet(**model_params).to(device)
    
    elif model_name == "DeepLabV3Plus":
        return DeepLabV3Plus(**model_params).to(device)
    
    elif model_name == "UnetPlusPlus":
        return UnetPlusPlus(**model_params).to(device)

    elif model_name == "SegFormer":
        return SegFormer(**model_params).to(device)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
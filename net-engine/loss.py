import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# MSELoss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input, target)

# 損失関数の取得
# def get_loss(loss_name):
#     if loss_name == 'DiceLoss':
#         return smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
#     elif loss_name == 'MSELoss':
#         return MSELoss()
#     else:
#         raise ValueError('Invalid loss name')

# 複合損失の追加
# 複合損失（Dice + BCE）
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, input, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]

        if target.dtype != input.dtype:
            target = target.type_as(input)

        return self.dice_weight * self.dice_loss(input, target) + self.bce_weight * self.bce_loss(input, target)


# 重み調整の変更
def get_loss(loss_name):
    if loss_name == 'DiceLoss':
        return smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    elif loss_name == 'MSELoss':
        return MSELoss()
    elif loss_name == 'DiceBCELoss':
        return DiceBCELoss(dice_weight=0.5, bce_weight=0.5)  # 重み調整可能
    else:
        raise ValueError('Invalid loss name')

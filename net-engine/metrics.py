import segmentation_models_pytorch as smp

# JSI (Jaccard Similarity Index) の計算関数
def jaccard_index(pred, target, threshold=0.5):
    """
    Jaccard Similarity Index (Intersection over Union, IoU)を計算する関数
    Args:
        pred (torch.Tensor): 予測されたマスク
        target (torch.Tensor): 正解のマスク
        threshold (float): 二値化のための閾値（デフォルトは0.5）
    Returns:
        float: JSI (IoU)
    """
    pred = (pred > threshold).float()
    intersection = (pred * target).sum().float()
    union = (pred + target).clamp(0, 1).sum().float()
    if union == 0:
        return float('nan')  # Unionが0ならばJSIは計算できない
    return (intersection / union).item()

def iou(pred, target, threshold=0.5):
    pred_mask = (pred > threshold).float()
    pred_mask = pred_mask.squeeze(1)

    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), target.long(), mode="binary")
    
    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

    return per_image_iou, dataset_iou
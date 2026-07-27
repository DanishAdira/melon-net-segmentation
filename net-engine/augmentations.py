import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

# データ拡張設定を取得する関数
def get_augmentation_train(augmentations: dict):
    albumentations_transforms = []
    torchvision_transforms = []

    # 中心50%を切り取る
    albumentations_transforms.append(A.Resize(512, 512))
    albumentations_transforms.append(A.CenterCrop(256, 256))

    """ Albumentation Data Augmentation
    Pixel-level transforms (https://explore.albumentations.ai/)
        * 実装済み, - 未実装
        - Advanced Blur
        * Blur
        - CLAHE
        - Channel Dropout
        - Channel Shuffle
        - Chromatic Aberration
        - Color Jitter
        - Defocus
        - Downscale
        - Emboss
        - Equalize
        - FDA
        - Fancy PCA
        - From Float
        * Gauss Noise
        - Gaussian Blur
        - Glass Blur
        - Histogram Matching
        * Hue Saturation Value
        - ISO Noise
        - Image Compression
        - Invert Img
        - Median Blur
        - Motion Blur
        - Multiplicative Noise
        - Normalize
        - Pixel distribution Adaptation
        - Planckian Jitter
        - Posterize
        * RGB Shift
        - Random Brightness Contrast
        - Random Fog
        - Random Gamma
        - Random Gravel
        - Random Rain
        - Random Shadow
        - Random Snow
        - Random Sun Flare
        - Random Tone Curve
        - Ringing Over Shoot
        - Sharpen
        - Shot Noise
        - Solarize
        - Spatter
        - Superpixels
        - Template Transform
        - Text Image
        - To Float
        - To Gray
        - To RGB
        - To Sepia
        - Unsharp Mask
        - Zoom blur
    """
    

    # ぼかし
    blur_cfg = augmentations.get("blur", {})
    if blur_cfg.get("enabled", False):
        blur_limit = blur_cfg.get("blur_limit", 3)
        albumentations_transforms.append(A.Blur(blur_limit=blur_limit, p=0.5))

    # ガウシアンノイズ
    noise_cfg = augmentations.get("gaussian_noise", {})
    if noise_cfg.get("enabled", False):
        var_limit = noise_cfg.get("var_limit", 6000)
        albumentations_transforms.append(A.GaussNoise(var_limit=(0, var_limit), p=0.5))

    # 色相飽和度値
    hsv_cfg = augmentations.get("hue_saturation_value", {})
    if hsv_cfg.get("enabled", False):
        hue_shift_limit = hsv_cfg.get("hue_shift_limit", 20)
        sat_shift_limit = hsv_cfg.get("sat_shift_limit", 30)
        val_shift_limit = hsv_cfg.get("val_shift_limit", 20)
        albumentations_transforms.append(A.HueSaturationValue(hue_shift_limit=hue_shift_limit, sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit, p=0.5))

    # RGBシフト
    rgb_cfg = augmentations.get("rgb_shift", {})
    if rgb_cfg.get("enabled", False):
        r_shift_limit = rgb_cfg.get("r_shift_limit", 20)
        g_shift_limit = rgb_cfg.get("g_shift_limit", 20)
        b_shift_limit = rgb_cfg.get("b_shift_limit", 20)
        albumentations_transforms.append(A.RGBShift(r_shift_limit=r_shift_limit, g_shift_limit=g_shift_limit, b_shift_limit=b_shift_limit, p=0.5))



    """ Albumentation Data Augmentation
    Spatial-level transforms (https://explore.albumentations.ai/)
        * 実装済み, - 未実装 
        - Affine
        - BBox Safe Random Crop
        - Center Crop
        - Coarse Dropout
        - Crop
        - Crop And Pad
        - Crop Non Empty Mask If Exists
        - D4
        * Elastic Transform
        * Grid Distortion
        - Grid Dropout
        - Grid Elastic Deform
        * Horizontal Flip
        - Lambda
        - Longest Max Size
        - Mask Dropout
        - Morphological
        - No Op
        * Optical Distortion
        - Overlay Elements
        - Pad If Needed
        - Perspective
        - Piecewise Affine
        - Pixel Dropout
        - Random Crop
        - Random Crop From Boarders
        - Random Grid Shuffle
        - Random Resized Crop
        - Random Rotate 90
        * Random Scale
        - Random Sized BBox Safe Crop
        - Random Sized Crop
        - Resize
        - Rotate
        - Safe Rotate
        * Shift Scale Rotate
        - Smallest Max Size
        - Transpose
        - Vertical Flip
        - XY Masking
    """


    # 弾性変形
    elastic_cfg = augmentations.get("elastic_transform", {})
    if elastic_cfg.get("enabled", False):
        alpha = elastic_cfg.get("alpha", 1)
        sigma = elastic_cfg.get("sigma", 10)
        albumentations_transforms.append(A.ElasticTransform(alpha=alpha, sigma=sigma, p=0.5))

    # グリッド変形
    grid_cfg = augmentations.get("grid_distortion", {})
    if grid_cfg.get("enabled", False):
        num_steps = grid_cfg.get("num_steps", 10)
        distort_limit = grid_cfg.get("distort_limit", 0.3)
        albumentations_transforms.append(A.GridDistortion(num_steps=num_steps, distort_limit=distort_limit, p=0.5))

    # 水平反転
    horizontal_flip_cfg = augmentations.get("horizontal_flip", {})
    if horizontal_flip_cfg.get("enabled", False):
        albumentations_transforms.append(A.HorizontalFlip(p=0.5))

    # オプティカルディストーション
    optical_cfg = augmentations.get("optical_distortion", {})
    if optical_cfg.get("enabled", False):
        distort_limit = optical_cfg.get("distort_limit", 0.3)
        albumentations_transforms.append(A.OpticalDistortion(distort_limit=distort_limit, p=0.5))

    # スケーリング
    scaling_cfg = augmentations.get("scaling", {})
    if scaling_cfg.get("enabled", False):
        min_scale = scaling_cfg.get("min_scale", 0.8)
        max_scale = scaling_cfg.get("max_scale", 1.2)
        albumentations_transforms.append(A.RandomScale(scale_limit=(min_scale, max_scale), p=0.5))

    # アフィン変換
    affine_cfg = augmentations.get("affine_transform", {})
    if affine_cfg.get("enabled", False):
        rotation_range = affine_cfg.get("rotation_range", 20)
        translate_range = affine_cfg.get("translate_range", 0.1)
        albumentations_transforms.append(A.ShiftScaleRotate(shift_limit=translate_range, scale_limit=0.1, rotate_limit=rotation_range, p=0.5))
    
    """ Torchvision Data Augmentation (https://pytorch.org/vision/main/transforms.html)
    Geometry
        - Resize
        - Scale Jitter
        - Random Shortest Size
        - Random Resize
        - Random Crop
        - Random Resized Crop
        - Random IoU Crop
        - Center Crop
        - Five Crop
        - Ten Crop
        - Random Horizontal Flip
        - Random Vertical Flip
        - Pad
        - Random Zoom Out
        - Random Rotation
        - Random Affine
        - Random Perspective
        - Elastic Transform
    Color
        - Color jitter
        - Random Channel Permutation
        - Random Photometric Distort
        - Gray Scale
        - RGB
        - Random Gray Scale
        - Gaussian Blur
        - GAuussian Noise
        - Random Invert
        - Random Posterize
        - Random Solarize
        - Random Adjust Sharpness
        - Random Autocontrast
        - Random Equalize
    Other
        * Random Erasing
        - CutMix
        - MixUp
    """

    # Random Erasing
    random_erasing_cfg = augmentations.get("random_erasing", {})
    if random_erasing_cfg.get("enabled", False):
        torchvision_transforms.append( v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False))

    # リサイズ/正規化/テンソル化
    albumentations_transforms.append(A.Resize(512, 512))
    albumentations_transforms.append(A.Normalize((0.5, ), (0.5, )))
    albumentations_transforms.append(ToTensorV2())
    
    # Albumentationとtorchvisionの変換を結合
    def combined_transform(image, mask):
        # Albumentationの変換を適用
        augmented = A.Compose(albumentations_transforms, is_check_shapes=False)(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

        # torchvisionの変換を適用
        for t in torchvision_transforms:
            image = t(image)

        return {"image": image, "mask": mask}
    
    return combined_transform

def get_augmentation_validation():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize((0.5, ), (0.5, )),
        A.CenterCrop(256, 256),
        ToTensorV2()
    ], is_check_shapes=False)
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore")

class UNet(smp.Unet):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation=None):
        super(UNet, self).__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=classes, activation=activation)

    def forward(self, x):
        return super().forward(x)
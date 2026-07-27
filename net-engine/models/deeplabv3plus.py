import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore")

class DeepLabV3Plus(smp.DeepLabV3Plus):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation=None):
        super(DeepLabV3Plus, self).__init__(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            classes=classes, 
            activation=activation
        )

    def forward(self, x):
        return super().forward(x)
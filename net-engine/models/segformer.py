import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore")

class SegFormer(smp.Segformer):
    def __init__(self, encoder_name='mit_b0', encoder_weights='imagenet', classes=1, activation=None):
        super(SegFormer, self).__init__(
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            classes=classes, 
            activation=activation
        )

    def forward(self, x):
        return super().forward(x)

TRAIN_MODES = [
    "baseline",
    "reweight",
    #"individual",
]

DATASETS = [
    "HAM",
    "BCN",
]

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
]

MODEL_REGISTRY = {
    "ResNet50": "resnet50.a1_in1k",
    "MobileNetV2": "mobilenetv2_100.ra_in1k",
    "SwinT": "swin_tiny_patch4_window7_224.ms_in1k",
}

SEEDS = [7, 29, 71, 164, 190]

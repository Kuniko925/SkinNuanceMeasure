# config.py

TRAIN_MODES = [
    "baseline",
    "reweight",
    "regularization",
    "individual",
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

SEEDS = [
    12,
    123,
    1234,
]

MODEL_TYPES = {
    "MobileNetV3": "CNN",
    "ConvNext": "CNN",
    "FastVit": "Hybrid",
    "MobileVitV2": "Hybrid",
    "VitTiny": "ViT",
    "DeitTiny": "ViT",
}
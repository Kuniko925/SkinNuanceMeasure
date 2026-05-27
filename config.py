# config.py

TRAIN_MODES = [
    "baseline",
    "reweight",
    "regularization",
    "individual",
]

DB_NAMES = [
    "UTKFace",
    "milk10k/dermoscopic",
    "milk10k/clinical",
]

MODEL_TYPES = {
    "MobileNetV3": "CNN",
    "ConvNext": "CNN",
    "FastVit": "Hybrid",
    "MobileVitV2": "Hybrid",
    "VitTiny": "ViT",
    "DeitTiny": "ViT",
}
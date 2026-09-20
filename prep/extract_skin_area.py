from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =========================================================
# Dataset
# =========================================================
class SkinLesionDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted(
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocess(image)

        return idx, image, str(img_path)


# =========================================================
# DeepLabV3
# =========================================================
class DeepLabV3(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()

        self.base_model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        )

        self.base_model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_class,
            kernel_size=1
        )

    def forward(self, x):
        x = self.base_model(x)
        return x["out"]


# =========================================================
# Segmentation + lesion removal
# =========================================================
def process_dataset(dataset_name, model, batch_size=16):

    input_dir = PROJECT_ROOT / "skin" / dataset_name / "data"
    mask_dir = PROJECT_ROOT / "skin" / dataset_name / "mask"
    skin_dir = PROJECT_ROOT / "skin" / dataset_name / "skin"

    mask_dir.mkdir(parents=True, exist_ok=True)
    skin_dir.mkdir(parents=True, exist_ok=True)

    dataset = SkinLesionDataset(input_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n{dataset_name}")
    print(f"Images: {len(dataset)}")

    model.eval()

    with torch.no_grad():

        for _, images, paths in loader:

            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            # [B, 1, H, W] -> [B, H, W]
            masks = (
                probabilities > 0.5
            ).squeeze(1).cpu().numpy()

            for mask, img_path in zip(masks, paths):

                img_path = Path(img_path)
                image = cv2.imread(str(img_path))

                if image is None:
                    print(f"Failed: {img_path}")
                    continue

                image = cv2.resize(image, (224, 224))

                mask = mask.astype("uint8")

                # -----------------------------------------
                # 1. Binary lesion mask
                # -----------------------------------------
                mask_save_path = (
                    mask_dir /
                    f"{img_path.stem}.png"
                )

                cv2.imwrite(
                    str(mask_save_path),
                    mask * 255
                )

                # -----------------------------------------
                # 2. Lesion removed RGB image
                # -----------------------------------------
                skin_image = image.copy()

                # lesion = 1
                skin_image[mask == 1] = 0

                skin_save_path = (
                    skin_dir /
                    f"{img_path.stem}.png"
                )

                cv2.imwrite(
                    str(skin_save_path),
                    skin_image
                )

    print(f"{dataset_name}: completed")


# =========================================================
# Main
# =========================================================
def main():

    model_path = (
        PROJECT_ROOT /
        "skin" /
        "seg_model.pt"
    )

    model = DeepLabV3(num_class=1)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)

    for dataset_name in ["HAM", "BCN"]:
        process_dataset(
            dataset_name,
            model,
            batch_size=16
        )


if __name__ == "__main__":
    main()
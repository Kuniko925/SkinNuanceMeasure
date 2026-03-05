import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import cv2
from torch.utils.data import DataLoader
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent

class ModelEvaluator:
    def evaluate(self, model, loader):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for idx, images in loader:
                images = images.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                outputs = (outputs > 0.5).astype("uint8")
                preds.extend(outputs)
        return preds


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx]["filepath"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        # PyTorch Tensor
        image = preprocess(image)

        return idx, image


class DeepLabV3(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.base_model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_class,
            kernel_size=1
        )
    def forward(self, x):
        x = self.base_model(x)
        return x["out"]

def main():
    filepaths = [PROJECT_ROOT / "skin/milk10k/clinical/df_train.csv", PROJECT_ROOT / "skin/milk10k/clinical/df_valid.csv", PROJECT_ROOT / "skin/milk10k/clinical/df_test.csv"]
    for filepath in filepaths:

        df = pd.read_csv(filepath)
        data = SkinLesionDataset(df)
        loader = DataLoader(data, batch_size=16, shuffle=False)

        model_save_path = PROJECT_ROOT / "skin/milk10k/seg/model_19.pt"
        seg_model = DeepLabV3(num_class=1)
        seg_model.load_state_dict(torch.load(model_save_path))

        evaluator = ModelEvaluator()
        preds = evaluator.evaluate(seg_model, loader)

        out_dir = PROJECT_ROOT / "skin/milk10k/clinical/mask/"
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx_in_dataset in range(len(df)):
            img_path = df.iloc[idx_in_dataset]["filepath"]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_mask = preds[idx_in_dataset]
            lesion_mask = (pred_mask == 1)

            lesion_black = img.copy()
            lesion_black[lesion_mask] = 0
            save_path = out_dir / f"{idx_in_dataset}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(lesion_black, cv2.COLOR_RGB2BGR))

        print(f"Saved masks")

if __name__== "__main__":

    main()
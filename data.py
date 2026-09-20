import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TransDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["binary_label_id"])
        skin_tone = int(row["skin_tone_group_id"])

        wd = int(row["wd"]) if "wd" in row.index else 0

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(skin_tone, dtype=torch.long),
            torch.tensor(wd, dtype=torch.long),
        )


class LoadDataset:
    def __init__(
        self,
        df,
        batch_size=32,
        image_size=(224, 224),
        isTrain=True,
        seed=42,
    ):
        self.df = df.copy()
        self.batch_size = batch_size
        self.image_size = image_size
        self.isTrain = isTrain
        self.seed = seed

    def get_dataloader(self):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.isTrain:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        dataset = TransDataset(
            self.df,
            transform=transform
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.isTrain,
            num_workers=4,
            generator=generator,
            worker_init_fn=seed_worker,
        )
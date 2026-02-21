import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TransDataset(Dataset):
    def __init__(self, dataframe, img_size, transform=None):
        self.dataframe = dataframe
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["filepath"]
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]["label"]
        image = self.transform(image) if self.transform else image
        return image, torch.tensor(label, dtype=torch.long)


class LoadDataset:
    def __init__(self, df, label, label_encoder, batch_size=32, image_size=(224, 224), isTrain=True):
        self.df = df.copy()
        self.label = label
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_encoder = label_encoder
        self.isTrain = isTrain

        if self.isTrain:
            self.df[self.label] = self.label_encoder.fit_transform(self.df[self.label])
        else:
            self.df[self.label] = self.label_encoder.transform(self.df[self.label])

    def get_dataloader(self):
        if self.isTrain:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])

        dataset = TransDataset(self.df, self.image_size, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.isTrain, num_workers=4)

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np

class KannadaMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.data_frame.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)
        image = np.array(Image.fromarray(image).convert('RGB'))
        label = self.data_frame.iloc[idx, 0]
        if self.transform:
          image = self.transform(image)

        return image, label
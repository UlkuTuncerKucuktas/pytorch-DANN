
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np

class KannadaMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = train_data = np.load('/content/pytorch-DANN/X_kannada_MNIST_test.npz')['arr_0']
        self.label = np.load('/content/pytorch-DANN/y_kannada_MNIST_test.npz')['arr_0']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.uint8)
        label = self.label[idx]
        if self.transform:
          image = self.transform(image)
        
        if image.shape[0] == 1:
                image = np.concatenate((image, image, image),axis=0)
        

        return image, label
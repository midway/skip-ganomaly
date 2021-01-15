import pandas as pd
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image, ImageOps

class MedCityDataset(Dataset):
    """MedCity dataset."""

    def __init__(self, csv_file, root, transform=None, image_size=64):
        """
        Args:
            csv_file (string): Path to the csv file with images.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_image_rows = pd.read_csv(csv_file)
        self.root_dir = root
        self.transform = transform
        self.image_size = image_size
        
        self.cached_data = {}
    def __len__(self):
        return len(self.dataset_image_rows)

    def get_cache_size(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx in self.cached_data:
            return self.cached_data[idx]

        if self.image_size is not None:
            img_name = os.path.join(f"{self.root_dir}{self.image_size}x{self.image_size}", self.dataset_image_rows.iloc[idx, 1])
        else:
            img_name = os.path.join(self.root_dir, self.dataset_image_rows.iloc[idx, 1])

        category = self.dataset_image_rows.iloc[idx, 2]
        if category in [ 1.0, 2.0 ]:
            target = 0
        else:
            target = 1
        image = Image.open(img_name, 'r').convert('L')

        if self.transform:
            image = self.transform(image)

        self.cached_data[idx] = ( image, target )

        return ( image, target )

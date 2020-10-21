import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image

class MedCityDataset(Dataset):
    """MedCity dataset."""

    def __init__(self, csv_file, root, transform=None):
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

    def __len__(self):
        return len(self.dataset_image_rows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataset_image_rows.iloc[idx, 1])
        category = self.dataset_image_rows.iloc[idx, 2]
        if category in [ '1.0', '2.0' ]:
            target = 0
        else:
            target = 1
        image = Image.fromarray(io.imread(img_name))

        if self.transform:
            image = self.transform(image)

        return ( image, target )
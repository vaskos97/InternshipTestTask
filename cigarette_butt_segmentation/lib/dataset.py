from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from lib.utils import get_mask
import json


class BasicDataset(Dataset):
    def __init__(self, path, annotations):
        self.path = path
        self.annotations = annotations

        self.ids = [splitext(file)[0] for file in listdir(path)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, i):
        idx = int(self.ids[i])

        img = np.array(Image.open(f"{self.path}/{idx:08}.jpg"))
        img = img.transpose((2, 0, 1))
        mask = get_mask(idx, self.annotations)
        

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

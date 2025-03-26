import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class H5UnalignedDataset(Dataset):
    def __init__(self, h5_path_A, h5_path_B, transform=None):
        super().__init__()
        self.h5_path_A = h5_path_A
        self.h5_path_B = h5_path_B
        self.transform = transform

        with h5py.File(self.h5_path_A, 'r') as fA:
            self.keys_A = list(fA.keys())
        with h5py.File(self.h5_path_B, 'r') as fB:
            self.keys_B = list(fB.keys())

        self.len_A = len(self.keys_A)
        self.len_B = len(self.keys_B)
        self.length = max(self.len_A, self.len_B)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index_A = idx % self.len_A
        index_B = idx % self.len_B

        with h5py.File(self.h5_path_A, 'r') as fA:
            img_A = torch.tensor(fA[self.keys_A[index_A]]['img'][()])  # [3, H, W]
        with h5py.File(self.h5_path_B, 'r') as fB:
            img_B = torch.tensor(fB[self.keys_B[index_B]]['img'][()])  # [3, H, W]

        # Assure toi qu'ils sont bien de type float et entre 0-1
        img_A = img_A.float() / 255.0
        img_B = img_B.float() / 255.0

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B, 'A_paths': self.keys_A[index_A], 'B_paths': self.keys_B[index_B]}

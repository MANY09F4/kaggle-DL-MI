import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class H5UnalignedDataset(Dataset):
    def __init__(self, h5_path_A, h5_path_B, transform=None, max_items_A=None, max_items_B=None, seed=42, test=False):
        super().__init__()
        self.h5_path_A = h5_path_A if isinstance(h5_path_A, list) else [h5_path_A]
        self.h5_path_B = h5_path_B if isinstance(h5_path_B, list) else [h5_path_B]
        self.transform = transform
        self.seed = seed

        random.seed(self.seed)

        self.keys_A = self._load_balanced_keys(self.h5_path_A, max_items_A)
        self.keys_B = self._load_balanced_keys(self.h5_path_B, max_items_B)

        self.len_A = len(self.keys_A)
        self.len_B = len(self.keys_B)
        self.length = max(self.len_A, self.len_B)

    def _load_balanced_keys(self, h5_paths, max_items):
        keys = []
        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    label = int(np.array(f[key]['label']))
                    keys.append((path, key, label))

        if max_items is None:
            return keys

        class_0 = [k for k in keys if k[2] == 0]
        class_1 = [k for k in keys if k[2] == 1]
        n_per_class = min(max_items // 2, len(class_0), len(class_1))

        selected = random.sample(class_0, n_per_class) + random.sample(class_1, n_per_class)
        random.shuffle(selected)
        return selected

    def __getitem__(self, idx):
        index_A = idx % self.len_A
        index_B = idx % self.len_B

        path_A, key_A, _ = self.keys_A[index_A]
        path_B, key_B, _ = self.keys_B[index_B]

        with h5py.File(path_A, 'r') as fA, h5py.File(path_B, 'r') as fB:
            img_A = torch.tensor(fA[key_A]['img'][()])
            img_B = torch.tensor(fB[key_B]['img'][()])

        if img_A.ndim == 3 and img_A.shape[-1] == 3:
            img_A = img_A.permute(2, 0, 1)
        if img_B.ndim == 3 and img_B.shape[-1] == 3:
            img_B = img_B.permute(2, 0, 1)

        img_A = img_A.float() * 2.0 - 1.0
        img_B = img_B.float() * 2.0 - 1.0


        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {
            'A': img_A,
            'B': img_B,
            'A_paths': f"{path_A}:{key_A}",
            'B_paths': f"{path_B}:{key_B}"
        }


def count_labels_and_centers(h5_path, selected_keys):
    """Compte le nombre d'images par centre et par label dans un fichier .h5 donné"""
    stats = {}  # {centre: {label: count}}

    with h5py.File(h5_path, 'r') as f:
        for key in selected_keys:
            label = int(np.array(f[key]['label']))
            center = int(np.array(f[key]['metadata'])[0])
            if center not in stats:
                stats[center] = {0: 0, 1: 0}
            stats[center][label] += 1

    return stats

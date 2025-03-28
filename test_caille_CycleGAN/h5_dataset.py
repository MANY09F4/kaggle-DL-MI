import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class H5UnalignedDataset(Dataset):
    def __init__(self, h5_path_A, h5_path_B, transform=None, max_items_A=None, max_items_B=None, seed=42):
        super().__init__()
        self.h5_path_A = h5_path_A
        self.h5_path_B = h5_path_B
        self.transform = transform
        self.seed = seed

        random.seed(self.seed)

        # Sous-échantillonnage équilibré des données
        self.keys_A = self._load_balanced_keys(self.h5_path_A, max_items_A)
        self.keys_B = self._load_balanced_keys(self.h5_path_B, max_items_B)

        self.len_A = len(self.keys_A)
        self.len_B = len(self.keys_B)
        self.length = max(self.len_A, self.len_B)

    def _load_balanced_keys(self, h5_path, max_items):
        with h5py.File(h5_path, 'r') as f:
            keys_by_label = {0: [], 1: []}
            for key in f.keys():
                label = int(np.array(f[key]['label']))
                if label in keys_by_label:
                    keys_by_label[label].append(key)

        if max_items is None:
            return keys_by_label[0] + keys_by_label[1]

        random.shuffle(keys_by_label[0])
        random.shuffle(keys_by_label[1])

        n_per_class = min(max_items // 2, len(keys_by_label[0]), len(keys_by_label[1]))
        selected_keys = keys_by_label[0][:n_per_class] + keys_by_label[1][:n_per_class]
        random.shuffle(selected_keys)
        return selected_keys

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index_A = idx % self.len_A
        index_B = idx % self.len_B

        with h5py.File(self.h5_path_A, 'r') as fA:
            img_A = torch.tensor(fA[self.keys_A[index_A]]['img'][()])
        with h5py.File(self.h5_path_B, 'r') as fB:
            img_B = torch.tensor(fB[self.keys_B[index_B]]['img'][()])

        # Normalisation [0, 1]
        img_A = img_A.float() / 255.0
        img_B = img_B.float() / 255.0

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {
            'A': img_A,
            'B': img_B,
            'A_paths': self.keys_A[index_A],
            'B_paths': self.keys_B[index_B]
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

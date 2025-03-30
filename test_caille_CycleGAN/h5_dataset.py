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
        self.is_test = test

        random.seed(self.seed)

        self.keys_A = self._load_keys(self.h5_path_A, max_items_A, balance_labels=not self.is_test)
        self.keys_B = self._load_keys(self.h5_path_B, max_items_B, balance_labels=not self.is_test)

        self.len_A = len(self.keys_A)
        self.len_B = len(self.keys_B)

    def _load_keys(self, paths, max_items, balance_labels=True):
        keys = []

        for path in paths:
            with h5py.File(path, 'r') as f:
                all_keys = list(f.keys())

                if balance_labels:
                    keys_by_label = {0: [], 1: []}
                    for key in all_keys:
                        label = int(np.array(f[key]['label']))
                        if label in keys_by_label:
                            keys_by_label[label].append(key)

                    if max_items is None:
                        selected = keys_by_label[0] + keys_by_label[1]
                    else:
                        random.shuffle(keys_by_label[0])
                        random.shuffle(keys_by_label[1])
                        n = min(max_items // 2, len(keys_by_label[0]), len(keys_by_label[1]))
                        selected = keys_by_label[0][:n] + keys_by_label[1][:n]
                else:
                    selected = all_keys if max_items is None else random.sample(all_keys, min(max_items, len(all_keys)))

                keys.extend([(path, k) for k in selected])

        random.shuffle(keys)
        return keys

    def __len__(self):
        return max(len(self.keys_A), len(self.keys_B))

    def __getitem__(self, idx):
        index_A = idx % self.len_A
        index_B = idx % self.len_B

        path_A, key_A = self.keys_A[index_A]
        path_B, key_B = self.keys_B[index_B]

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
    stats = {}  # {centre: {label: count}}

    with h5py.File(h5_path, 'r') as f:
        for key in selected_keys:
            label = int(np.array(f[key]['label']))
            center = int(np.array(f[key]['metadata'])[0])
            if center not in stats:
                stats[center] = {0: 0, 1: 0}
            stats[center][label] += 1

    return stats

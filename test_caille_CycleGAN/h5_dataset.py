import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class H5UnalignedDataset(Dataset):
    def __init__(self, h5_path_A, h5_path_B, transform=None, max_items_A=None, max_items_B=None, seed=42, domain=None, aberrant_ids=None):
        super().__init__()
        self.h5_path_A = h5_path_A if isinstance(h5_path_A, list) else [h5_path_A]
        self.h5_path_B = h5_path_B if isinstance(h5_path_B, list) else [h5_path_B]
        self.transform = transform
        self.seed = seed
        self.domain = domain  # Domaine sélectionné
        self.aberrant_ids = aberrant_ids if aberrant_ids is not None else []  # Liste des images aberrantes à exclure

        random.seed(self.seed)

        # Si un domaine est spécifié, on le charge sinon on charge tous les centres
        if self.domain is None:
            self.keys_A = self._load_keys(self.h5_path_A, max_items_A, balance_labels=True)
        else:
            self.keys_A = self._load_keys(self.h5_path_A, max_items_A, balance_labels=True, domain=self.domain)

        self.keys_B = self._load_keys(self.h5_path_B, max_items_B, balance_labels=True)

        # Filtrer les clés aberrantes si nécessaire
        self.keys_A = [key for key in self.keys_A if key[1] not in self.aberrant_ids]
        self.keys_B = [key for key in self.keys_B if key[1] not in self.aberrant_ids]

        self.len_A = len(self.keys_A)
        self.len_B = len(self.keys_B)

    def _load_keys(self, paths, max_items, balance_labels=True, domain=None):
        keys = []

        for path in paths:
            with h5py.File(path, 'r') as f:
                all_keys = list(f.keys())

                if domain is not None:
                    # Si un domaine est sélectionné, on ne charge que les clés correspondantes à ce domaine
                    keys_by_center = {}
                    for key in all_keys:
                        center = int(np.array(f[key]['metadata'])[0])  # Récupérer le centre
                        if center == domain:
                            keys_by_center[key] = f[key]
                    selected = list(keys_by_center.keys())
                else:
                    # Sinon, on fait comme avant, on charge tous les centres
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

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class H5UnalignedDataset(Dataset):
    """
    Custom PyTorch Dataset for loading unaligned image pairs from two HDF5 datasets (domain A and domain B).
    Optionally supports domain-specific filtering and aberrant image removal.
    """

    def __init__(self, h5_path_A, h5_path_B, transform=None, max_items_A=None, max_items_B=None,
                 seed=42, domain=None, aberrant_ids_map=None):
        """
        Args:
            h5_path_A (str or list): Path(s) to HDF5 file(s) for source domain (A).
            h5_path_B (str or list): Path(s) to HDF5 file(s) for target domain (B).
            transform (callable, optional): Transformations to apply to both domains.
            max_items_A (int, optional): Max number of samples to use from domain A.
            max_items_B (int, optional): Max number of samples to use from domain B.
            seed (int): Random seed for reproducibility.
            domain (int, optional): Specific source domain ID to filter images (e.g., center 0).
            aberrant_ids_map (dict, optional): Dictionary mapping each file path to a list of image IDs to exclude.
        """
        super().__init__()
        self.h5_path_A = h5_path_A if isinstance(h5_path_A, list) else [h5_path_A]
        self.h5_path_B = h5_path_B if isinstance(h5_path_B, list) else [h5_path_B]
        self.transform = transform
        self.seed = seed
        self.domain = domain
        self.aberrant_ids_map = aberrant_ids_map if aberrant_ids_map is not None else {}

        random.seed(self.seed)

        if self.domain is None:
            self.keys_A = self._load_keys_source(self.h5_path_A, max_items_A, balance_labels=True)
        else:
            self.keys_A = self._load_keys_source(self.h5_path_A, max_items_A, balance_labels=True, domain=self.domain)

        self.keys_B = self._load_keys_target(self.h5_path_B, max_items_B)

        self.len_A = len(self.keys_A)
        self.len_B = len(self.keys_B)

    def _filter_aberrant(self, path, keys):
        """
        Remove aberrant image IDs (e.g., black/white corrupted patches).
        """
        ids_to_exclude = self.aberrant_ids_map.get(path, [])
        return [key for key in keys if int(key[1]) not in ids_to_exclude]

    def _load_keys_source(self, paths, max_items, balance_labels=True, domain=None):
        """
        Load keys from source domain (A) while optionally balancing labels or filtering by center.
        """
        keys = []
        for path in paths:
            with h5py.File(path, 'r') as f:
                all_keys = list(f.keys())

                if domain is not None:
                    # Filter images belonging to a specific center/domain
                    keys_by_center = {k: f[k] for k in all_keys if int(np.array(f[k]['metadata'])[0]) == domain}
                    selected = list(keys_by_center.keys())
                else:
                    if balance_labels:
                        keys_by_label = {0: [], 1: []}
                        for key in all_keys:
                            try:
                                label = int(np.array(f[key]['label']))
                                if label in keys_by_label:
                                    keys_by_label[label].append(key)
                            except KeyError:
                                pass
                        if max_items is None:
                            selected = keys_by_label[0] + keys_by_label[1]
                        else:
                            random.shuffle(keys_by_label[0])
                            random.shuffle(keys_by_label[1])
                            n = min(max_items // 2, len(keys_by_label[0]), len(keys_by_label[1]))
                            selected = keys_by_label[0][:n] + keys_by_label[1][:n]
                    else:
                        selected = all_keys if max_items is None else random.sample(all_keys, min(max_items, len(all_keys)))

                key_pairs = [(path, k) for k in selected]
                key_pairs = self._filter_aberrant(path, key_pairs)
                keys.extend(key_pairs)

        random.shuffle(keys)
        return keys

    def _load_keys_target(self, paths, max_items):
        """
        Load keys from target domain (B).
        """
        keys = []
        for path in paths:
            with h5py.File(path, 'r') as f:
                all_keys = list(f.keys())
                selected = all_keys if max_items is None else random.sample(all_keys, min(max_items, len(all_keys)))
                keys.extend([(path, k) for k in selected])  # No filtering for B

        random.shuffle(keys)
        return keys

    def __len__(self):
        print(f"Number of source domain images: {len(self.keys_A)}")
        return max(len(self.keys_A), len(self.keys_B))

    def __getitem__(self, idx):
        """
        Load one unaligned image pair (A, B), normalize them to [-1, 1], and apply transformations.
        """
        index_A = idx % self.len_A
        index_B = idx % self.len_B

        path_A, key_A = self.keys_A[index_A]
        path_B, key_B = self.keys_B[index_B]

        with h5py.File(path_A, 'r') as fA, h5py.File(path_B, 'r') as fB:
            img_A = torch.tensor(fA[key_A]['img'][()])
            img_B = torch.tensor(fB[key_B]['img'][()])

        # Ensure channel-first format [C, H, W]
        if img_A.ndim == 3 and img_A.shape[-1] == 3:
            img_A = img_A.permute(2, 0, 1)
        if img_B.ndim == 3 and img_B.shape[-1] == 3:
            img_B = img_B.permute(2, 0, 1)

        # Normalize to [-1, 1]
        img_A = img_A.float() * 2.0 - 1.0
        img_B = img_B.float() * 2.0 - 1.0

        # Apply transforms if provided
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {
            'A': img_A,
            'B': img_B,
            'A_paths': f"{path_A}:{key_A}",
            'B_paths': f"{path_B}:{key_B}"
        }

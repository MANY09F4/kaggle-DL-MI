from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import torch
from tqdm import tqdm
import torchstain


class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode, max_samples=None, center_balanced=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode

        with h5py.File(self.dataset_path, 'r') as hdf:
            all_ids = list(hdf.keys())

            if max_samples is None:
                self.image_ids = all_ids
            else:
                if center_balanced:
                    # Regrouper les IDs par centre
                    center_dict = {}
                    for img_id in all_ids:
                        center = int(hdf[img_id].get("metadata")[0])
                        center_dict.setdefault(center, []).append(img_id)

                    # Calculer combien d’images par centre
                    n_centers = len(center_dict)
                    per_center = max_samples // n_centers

                    # Sous-échantillonnage équilibré
                    balanced_ids = []
                    for ids in center_dict.values():
                        if len(ids) < per_center:
                            print("Moins d’images que prévu pour un centre")
                        balanced_ids.extend(np.random.choice(ids, size=min(len(ids), per_center), replace=False))

                    np.random.shuffle(balanced_ids)
                    self.image_ids = balanced_ids
                else:
                    self.image_ids = np.random.choice(all_ids, size=max_samples, replace=False).tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(hdf[img_id]["img"])
            label = np.array(hdf[img_id].get("label")) if self.mode == 'train' else None
        return self.preprocessing(img).float(), label

def precompute(dataloader, model, device):
    xs, ys = [], []
    for x, y in tqdm(dataloader, leave=False):
        with torch.no_grad():
            xs.append(model(x.to(device)).detach().cpu().numpy())
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.hstack(ys)
    return torch.tensor(xs), torch.tensor(ys)

class PrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        super(PrecomputedDataset, self).__init__()
        self.features = features
        self.labels = labels.unsqueeze(-1)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()
    
def preprocess_base(type,feature_extractor,device,preprocessing,args):
    if type == "train":
        if args.size_train > 0:
            max_samples_train = args.size_train
            max_samples_val = int(args.size_train*0.3)
        else:
            max_samples_train =   None
            max_samples_val = None

        train_dataset = BaselineDataset(args.train_path, preprocessing, type, max_samples=max_samples_train)
        val_dataset = BaselineDataset(args.val_path, preprocessing, type, max_samples=max_samples_val)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

        train_dataset = PrecomputedDataset(*precompute(train_loader, feature_extractor, device))
        val_dataset = PrecomputedDataset(*precompute(val_loader, feature_extractor, device))

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
        return train_loader, val_loader
    else:
        return 0 


def apply_macenko(dataset,test_path,preprocessing_test,idx_test):

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    test_dataset = BaselineDataset(test_path, preprocessing_test, 'train')
    target_img = test_dataset[idx_test][0]
    normalizer.fit(target_img)


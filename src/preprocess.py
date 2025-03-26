from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import torch
from tqdm import tqdm
import torchvision.transforms as transforms


class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode):
        super(BaselineDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        
        with h5py.File(self.dataset_path, 'r') as hdf:        
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(hdf.get(img_id).get('img'))
            label = np.array(hdf.get(img_id).get('label')) if self.mode == 'train' else None
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
    
def preprocess_base(type,feature_extractor,device,args):
    if type == "train":  
        preprocessing = transforms.Resize((98, 98))
        train_dataset = BaselineDataset(args.train_path, preprocessing, type)
        val_dataset = BaselineDataset(args.val_path, preprocessing, type)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

        train_dataset = PrecomputedDataset(*precompute(train_loader, feature_extractor, device))
        val_dataset = PrecomputedDataset(*precompute(val_loader, feature_extractor, device))

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
        return train_loader, val_loader
    else:
        return 0 

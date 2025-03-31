from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import torch
from tqdm import tqdm
import torchstain
import torchvision.transforms as transforms

class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode, stain_normalizer=None, max_samples=None, center_balanced=True):
        super().__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        self.stain_normalizer = stain_normalizer

        with h5py.File(dataset_path, 'r') as hdf:
            all_ids = list(hdf.keys())

            if max_samples is None:
                self.image_ids = all_ids
            else:
                if center_balanced:
                    center_dict = {}
                    for img_id in all_ids:
                        center = int(hdf[img_id].get("metadata")[0])
                        center_dict.setdefault(center, []).append(img_id)

                    n_centers = len(center_dict)
                    per_center = max_samples // n_centers

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
            img = np.array(hdf[img_id]["img"])
            label = np.array(hdf[img_id].get("label")) if self.mode == 'train' else None
        
        #print("1",img.shape,img.max())

        if img.shape[-1]==3:
            img = torch.tensor(img).permute(2, 0, 1).float()
        else:
            img = torch.tensor(img).float()

        if img.max() <= 1.0:
            img = img * 255.
        
        try:
            if self.stain_normalizer is not None:
                img, _, _ = self.stain_normalizer.normalize(img, stains=True)
                img = img.permute(2,1,0)
        except Exception as e:
            print(f"Stain normalization failed on idx={idx} | image shape={img.shape} | max={img.max()}")
            # Option 1 : retour image non normalisée (fallback)
            # Option 2 : choisir une image random dans le dataset
            img = img  # fallback : ne rien faire (on peut aussi remplacer par torch.zeros_like(img))

        
        #print("2",img.shape,img.max())

        img = self.preprocessing(img).float()

        if img.max() > 1.0:
            img = img / 255.0

        #print("3",img.shape,img.max())
        return img, label


def precompute(dataloader, model, device):
    xs, ys = [], []
    for x, y in tqdm(dataloader, leave=False):
        with torch.no_grad():
            #print(x.shape,x.max())
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


def preprocess_macenko(type,feature_extractor,device,preprocessing,idx_target_macenko,args):
    if type == "train":
        if args.size_train > 0:
            max_samples_train = args.size_train
            max_samples_val = int(args.size_train*0.3)
        else:
            max_samples_train =   None
            max_samples_val = None


        test_dataset = BaselineDataset(args.test_path, preprocessing=lambda x: x, mode="train")
        target_img, _ = test_dataset[idx_target_macenko]
        #target_img = target_img.permute((1,2,0))
        target_img = transforms.Resize((96, 96))(target_img)
        #print("test",target_img.shape,target_img.max())

        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        normalizer.fit(target_img)

 
        train_dataset = BaselineDataset(
            args.train_path, preprocessing, mode="train",
            max_samples=max_samples_train, stain_normalizer=normalizer
        )
        val_dataset = BaselineDataset(
            args.val_path, preprocessing, mode="train",
            max_samples=max_samples_val, stain_normalizer=normalizer
        )

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

        train_dataset = PrecomputedDataset(*precompute(train_loader, feature_extractor, device))
        val_dataset = PrecomputedDataset(*precompute(val_loader, feature_extractor, device))

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

        return train_loader, val_loader




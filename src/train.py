#Import 
import numpy as np
from tqdm import tqdm
import torch
import argparse
import torchmetrics
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms



def train_one_epoch(model, dataloader, optimizer, criterion, metric, device):
    model.train()
    metrics, losses = [], []
    for x, y in tqdm(dataloader, leave=False):
        optimizer.zero_grad()
        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
        loss.backward()
        optimizer.step()
        losses.extend([loss.item()] * len(y))
        score = metric(pred.cpu(), y.int().cpu())
        metrics.extend([score.item()] * len(y))
    return np.mean(losses), np.mean(metrics)

def validate(model, dataloader, criterion, metric, device):
    model.eval()
    metrics, losses = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, leave=False):
            pred = model(x.to(device))
            loss = criterion(pred, y.to(device))
            losses.extend([loss.item()] * len(y))
            score = metric(pred.cpu(), y.int().cpu())
            metrics.extend([score.item()] * len(y))
    return np.mean(losses), np.mean(metrics)



# ==== Fonction principale ====
def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.preprocessing == "base":
        preprocessing = transforms.Resize((98, 98))
    elif args.preprocessing == "grey":
        preprocessing = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Grayscale(num_output_channels=3),  
                            transforms.Resize((98, 98)),
                            transforms.ToTensor()
                        ])


    # ==== Données ====
    if args.preprocess_type == "base":
        from preprocess import preprocess_base
        feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        feature_extractor.eval()
        train_loader, val_loader = preprocess_base(type="train",feature_extractor=feature_extractor,preprocessing=preprocessing,device=device,args=args) 
    elif args.preprocess_type == "macenko":
        from preprocess import preprocess_macenko
        feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        feature_extractor.eval()
        train_loader, val_loader = preprocess_macenko(type="train",feature_extractor=feature_extractor,preprocessing=preprocessing,device=device,args=args,idx_target_macenko=2) 

    
    # ==== Modèle ====
    if args.model == 'base':
        from models import linear_probing
        model = linear_probing(feature_extractor=feature_extractor).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")


    # ==== Optimiseur, loss, metric ====
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    criterion = getattr(torch.nn, args.loss)()
    metric = getattr(torchmetrics, args.metric)('binary')

    min_loss, best_epoch = float('inf'), 0

    for epoch in range(args.epochs):
        train_loss, train_metric = train_one_epoch(model, train_loader, optimizer, criterion, metric, device)
        val_loss, val_metric = validate(model, val_loader, criterion, metric, device)

        print(f"[{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | Acc: {train_metric:.4f}")
        print(f"[{epoch+1}/{args.epochs}] Val   Loss: {val_loss:.4f} | Acc: {val_metric:.4f}")

        if val_loss < min_loss:
            print(f"New best val loss {min_loss:.4f} → {val_loss:.4f}")
            min_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.checkpoint_path)

        if epoch - best_epoch >= args.patience:
            print("Early stopping.")
            break


# ==== Argument parser ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base", help="Model type")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--checkpoint_path", type=str, default='../checkpoints/models/best_model.pth')
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer type")
    parser.add_argument("--loss", type=str, default='BCELoss', help="Loss type")
    parser.add_argument("--metric", type=str, default='Accuracy', help="Metric type")
    parser.add_argument("--preprocess_type", type=str, default='base', help="Preprocessing type")
    parser.add_argument("--train_path", type=str, default='../data/train.h5', help="Data Train Path")
    parser.add_argument("--val_path", type=str, default='../data/val.h5', help="Data Validation Path")
    parser.add_argument("--test_path", type=str, default='../data/test.h5', help="Data Test Path")
    parser.add_argument("--preprocessing", type=str, default='base', help="type of preprocessing")
    parser.add_argument("--size_train", type=int, default=0)

    args = parser.parse_args()

    main(args)


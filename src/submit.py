# submit.py

import argparse
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms

from src.models import linear_probing


def load_model(model_type, checkpoint_path, feature_extractor, device):
    if model_type == 'base':
        model = linear_probing(feature_extractor).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load feature extractor ===
    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    feature_extractor.eval()

    # === Load model ===
    model = load_model(args.model, args.checkpoint, feature_extractor, device)

    # === Preprocessing (must match training) ===
    preprocessing = transforms.Resize((98, 98))

    # === Predictions ===
    solutions_data = {'ID': [], 'Pred': []}

    with h5py.File(args.test_path, 'r') as hdf:
        test_ids = list(hdf.keys())
        for test_id in tqdm(test_ids):
            img = np.array(hdf[test_id]['img'])
            img_tensor = preprocessing(torch.tensor(img)).unsqueeze(0).float().to(device)

            with torch.no_grad():
                features = feature_extractor(img_tensor)
                pred = model(features).cpu().item()

            solutions_data['ID'].append(int(test_id))
            solutions_data['Pred'].append(int(pred > 0.5))  

    # === Save submission ===
    df = pd.DataFrame(solutions_data).set_index('ID')
    df.to_csv(args.output)
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='base', help='Model type (e.g. base)')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='../data/test.h5', help='Path to test H5 file')
    parser.add_argument('--output', type=str, default='../checkpoints/submit/submit.csv', help='Output CSV path for submission')
    args = parser.parse_args()

    main(args)

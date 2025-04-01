# submit.py

import argparse
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

from models import linear_probing
import sys
import os

# Ajouter le dossier "projet" au path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # remonte à "projet"
sys.path.append(ROOT_DIR)
from test_caille_CycleGAN.util.util import save_image

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
    if args.preprocessing == "base":
        preprocessing = transforms.Resize((98, 98))
    elif args.preprocessing == "grey":  
        preprocessing = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Grayscale(num_output_channels=3),  
                        transforms.Resize((98, 98)),
                        transforms.ToTensor()
                    ])


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

            if test_id == test_ids[2]:
                img2 = np.array(hdf[test_ids[2]]['img'])
                if img2.shape[0] == 3:  # (C, H, W)
                    img2 = np.transpose(img2, (1, 2, 0))
                    img2 = (img2 * 255).astype(np.uint8) 

                save_image(img2,"checkpoints/pics/image_test_2.png")



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
    parser.add_argument("--preprocessing", type=str, default='base', help="type of preprocessing")
    args = parser.parse_args()

    main(args)

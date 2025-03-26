#Import
import torch 

def linear_probing(feature_extractor):
    return torch.nn.Sequential(torch.nn.Linear(feature_extractor.num_features, 1),
                                torch.nn.Sigmoid())
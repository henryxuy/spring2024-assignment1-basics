import torch

def softmax_implemented(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    nominator = torch.exp(in_features - torch.max(in_features, dim=dim, keepdim=True)[0])
    denominator = torch.sum(nominator, dim=dim, keepdim=True)
    return nominator / denominator




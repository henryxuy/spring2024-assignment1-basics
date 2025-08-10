import torch

class RMSNormImplemented(torch.nn.Module):
    def __init__(self, d_model: int, weights: dict[str, torch.FloatTensor], eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(weights['weight'])
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_a = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / self.d_model + self.eps)
        output = self.weight * x / rms_a
        return output

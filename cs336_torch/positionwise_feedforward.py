import math
import torch

def gelu_implemented(in_features: torch.FloatTensor) -> torch.FloatTensor:
    return in_features * 0.5 * (1 + torch.erf(in_features / math.sqrt(2)))


class PositionwiseFeedforwardImplemented(torch.nn.Module):
    """Given the weights of a position-wise feedforward network, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are `w1.weight` and `w2.weight`.
            `w1` is the first linear transformation, and `w2` is the second
            linear transformation (eq. 2 of Vaswani et al., 2017).
            `w1.weight` is of shape (d_ff, d_model).
            `w2.weight` is of shape (d_model, d_ff).
    )
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your position-wise feedforward network
        with the provided `weights` on the provided `in_features`.
    """
    def __init__(self, d_model: int, d_ff: int, weights: dict[str, torch.FloatTensor]):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)
        self.w1.weight.data = weights['w1.weight']
        self.w2.weight.data = weights['w2.weight']

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        return self.w2(gelu_implemented(self.w1(in_features)))

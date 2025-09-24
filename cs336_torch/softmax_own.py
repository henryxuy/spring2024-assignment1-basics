import torch

def softmax_implemented(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """
    Numerically stable softmax implementation.
    
    Mathematical Formula:
    For input vector x = [x₁, x₂, ..., xₙ], softmax is defined as:
    
    softmax(xᵢ) = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
    
    Where:
    - xᵢ is the i-th element of the input
    - max(x) is the maximum value in the input (for numerical stability)
    - Σⱼ represents the sum over all elements j in the vector
    
    The subtraction of max(x) prevents overflow/underflow issues while 
    maintaining mathematical equivalence to the standard formula:
    softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
    
    Args:
        in_features: Input tensor to apply softmax to
        dim: Dimension along which to apply softmax
    
    Returns:
        Tensor with softmax applied along the specified dimension
    """
    # Numerator: exp(xᵢ - max(x)) - numerically stable exponential
    nominator = torch.exp(in_features - torch.max(in_features, dim=dim, keepdim=True)[0])
    
    # Denominator: Σⱼ exp(xⱼ - max(x)) - sum of all exponentials
    denominator = torch.sum(nominator, dim=dim, keepdim=True)
    
    # Final result: exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
    return nominator / denominator




import torch
from collections.abc import Iterable


def run_gradient_clipping_implemented(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    # Filter parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if not params_with_grad:
        return None  # No gradients to clip
    
    # Calculate total L2 norm of all gradients
    total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in params_with_grad))
    
    # Only clip if the total norm exceeds the maximum allowed norm
    if total_norm > max_l2_norm:
        clip_coeff = max_l2_norm / total_norm
        for p in params_with_grad:
            p.grad.data.mul_(clip_coeff)
    
    return None


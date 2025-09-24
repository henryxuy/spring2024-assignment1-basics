import torch
from cs336_torch.softmax_own import softmax_implemented

def cross_entropy_implemented(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    # Numerically stable log-softmax computation:
    # log(softmax(x_i)) = x_i - log(sum(exp(x_j))) = x_i - logsumexp(x)
    # This avoids computing softmax explicitly, preventing underflow issues
    
    # Compute log-softmax in a numerically stable way
    log_sum_exp = torch.logsumexp(inputs, dim=1, keepdim=True)  # (batch_size, 1)
    log_softmax = inputs - log_sum_exp  # (batch_size, num_classes)
    
    # Extract log probabilities for target classes
    log_target_probs = log_softmax[torch.arange(targets.shape[0]), targets]
    
    # Calculate cross-entropy loss = -1/N * sum(log(p_target_class))
    loss = -torch.mean(log_target_probs)
    
    return loss

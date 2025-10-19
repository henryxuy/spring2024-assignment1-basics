import torch
import numpy.typing as npt
import numpy as np


def get_batch_implemented(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    max_start_index = len(dataset) - context_length - 1
    starting_indices = np.random.randint(0, max_start_index + 1, batch_size)

    # Create contiguous sequences
    sequences = []
    for start_index in starting_indices:
        sequence = dataset[start_index:start_index + context_length]
        sequences.append(sequence)
    
    x = torch.tensor(np.array(sequences), dtype=torch.long, device=device)
    y = torch.tensor(np.array([dataset[start_index + 1:start_index + context_length + 1] 
        for start_index in starting_indices]), dtype=torch.long, device=device)

    return x, y
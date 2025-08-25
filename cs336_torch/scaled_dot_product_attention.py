import torch
from typing import Optional
import math
from cs336_torch.softmax_own import softmax_implemented


def scaled_dot_product_attention_implemented(K: torch.FloatTensor, Q: torch.FloatTensor, V: torch.FloatTensor, 
                                             mask: Optional[torch.BoolTensor] = None, 
                                             pdrop: Optional[float] = None) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    Q_K_t = Q @ K.transpose(-2, -1)
    d_k = Q.shape[-1]
    Q_K_t = Q_K_t / math.sqrt(d_k)

    if mask is not None:
        Q_K_t = Q_K_t.masked_fill(mask, float("-inf"))

    if pdrop is not None:
        Q_K_t = torch.nn.functional.dropout(Q_K_t, p=pdrop)

    return softmax_implemented(Q_K_t, dim=-1) @ V



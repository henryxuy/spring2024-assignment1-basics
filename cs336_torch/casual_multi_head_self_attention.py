import torch
from cs336_torch.scaled_dot_product_attention import scaled_dot_product_attention_implemented

def multihead_self_attention_implemented(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    # Create causal mask - upper triangular matrix to prevent attention to future tokens
    # For seq_len=4: [[F,T,T,T], [F,F,T,T], [F,F,F,T], [F,F,F,F]]
    # True positions will be masked out (set to -inf before softmax)
    causal_mask = torch.triu(torch.ones(in_features.shape[1], in_features.shape[1]), diagonal=1)
    causal_mask = causal_mask.bool()
    
    # BLOCKED MATRIX OPTIMIZATION:
    # Instead of processing each head separately (naive approach), we use blocked matrix multiplication.
    # Concatenate all head weights vertically to create block matrices:
    # q_weights = [q_heads.0.weight]  ← Block 0 (d_head × d_model)
    #             [q_heads.1.weight]  ← Block 1 (d_head × d_model)
    #             [      ...       ]
    # Final shape: (num_heads * d_head, d_model) = (d_model, d_model) when d_head = d_model/num_heads
    q_weights = torch.cat([weights[f"q_heads.{i}.weight"] for i in range(num_heads)], dim=0)
    k_weights = torch.cat([weights[f"k_heads.{i}.weight"] for i in range(num_heads)], dim=0)
    v_weights = torch.cat([weights[f"v_heads.{i}.weight"] for i in range(num_heads)], dim=0)
    
    # BLOCKED MATRIX MULTIPLICATION:
    # Single matrix operation computes ALL heads simultaneously instead of num_heads separate operations
    # (batch, seq_len, d_model) @ (d_model, num_heads*d_head) → (batch, seq_len, num_heads*d_head)
    # This is mathematically equivalent to individual head computations but much more efficient
    q_all = in_features @ q_weights.transpose(-2, -1)  # (batch, seq_len, d_head * num_heads)
    k_all = in_features @ k_weights.transpose(-2, -1)  # (batch, seq_len, d_head * num_heads)
    v_all = in_features @ v_weights.transpose(-2, -1)  # (batch, seq_len, d_head * num_heads)
    
    # RESHAPE TO RECOVER INDIVIDUAL HEADS:
    # Convert from blocked output back to multi-head structure
    # (batch, seq_len, num_heads*d_head) → (batch, seq_len, num_heads, d_head)
    batch_size, seq_len = in_features.shape[:2]
    d_head = q_all.shape[-1] // num_heads
    
    q_all = q_all.view(batch_size, seq_len, num_heads, d_head)
    k_all = k_all.view(batch_size, seq_len, num_heads, d_head)  
    v_all = v_all.view(batch_size, seq_len, num_heads, d_head)
    
    # TRANSPOSE FOR PARALLEL ATTENTION:
    # Move heads to batch dimension for parallel processing: (batch, num_heads, seq_len, d_head)
    # This allows scaled_dot_product_attention to process all heads simultaneously
    q_all = q_all.transpose(1, 2)
    k_all = k_all.transpose(1, 2)
    v_all = v_all.transpose(1, 2)
    
    # Apply scaled dot-product attention to ALL heads in parallel
    # Each head computes: softmax(QK^T / √d_k) V with the same causal mask
    attn_output = scaled_dot_product_attention_implemented(k_all, q_all, v_all, causal_mask, attn_pdrop)
    
    # CONCATENATE HEAD OUTPUTS:
    # Transpose back: (batch, num_heads, seq_len, d_head) → (batch, seq_len, num_heads, d_head)
    # Then flatten heads: (batch, seq_len, num_heads, d_head) → (batch, seq_len, num_heads*d_head)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch, seq_len, num_heads * d_head)
    
    # FINAL OUTPUT PROJECTION:
    # Apply W^O matrix from Transformer paper. Key fix: transpose the weight matrix!
    # output_proj.weight shape: (d_model, num_heads*d_head)
    # Need: (batch, seq_len, num_heads*d_head) @ (num_heads*d_head, d_model) → (batch, seq_len, d_model)
    return attn_output @ weights["output_proj.weight"].transpose(-2, -1)

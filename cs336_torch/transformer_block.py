import torch
from cs336_torch.casual_multi_head_self_attention import multihead_self_attention_implemented
from cs336_torch.rmsnorm import RMSNormImplemented
from cs336_torch.softmax_own import softmax_implemented
from cs336_torch.positionwise_feedforward import PositionwiseFeedforwardImplemented

def transformer_block_implemented(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Args:
        d_model: int
            The dimensionality of the Transformer block input.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the output of each sub-layer, before it
            is added to the sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, (d_model / num_heads) * num_heads).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
            Shape is (batch_size, sequence_length, d_model).

    Returns:
        FloatTensor of shape (batch_size, sequence_length, d_model) with the output of
        running the Transformer block on the input features.
    """

    # Pre-norm Transformer Block Architecture:
    # y = x + Dropout(MultiHeadSelfAttention(RMSNorm(x)))
    # z = y + Dropout(FFN(RMSNorm(y)))

    # FIRST SUBLAYER: MultiHeadSelfAttention with pre-norm
    # Step 1: Apply first RMSNorm
    ln1_weights = {'weight': weights['ln1.weight']}
    rmsnorm1 = RMSNormImplemented(d_model, ln1_weights, eps=1e-5)
    rmsnorm1_output = rmsnorm1(in_features)
    
    # Step 2: Apply MultiHeadSelfAttention
    # Convert consolidated weights to individual head format expected by multihead_self_attention_implemented
    d_head = d_model // num_heads
    attn_weights = {}
    
    # Split consolidated attn.q_proj.weight into individual q_heads.{i}.weight
    q_proj_weight = weights['attn.q_proj.weight']  # Shape: (num_heads * d_head, d_model)
    for i in range(num_heads):
        start_idx = i * d_head
        end_idx = (i + 1) * d_head
        attn_weights[f'q_heads.{i}.weight'] = q_proj_weight[start_idx:end_idx, :]
    
    # Split consolidated attn.k_proj.weight into individual k_heads.{i}.weight  
    k_proj_weight = weights['attn.k_proj.weight']  # Shape: (num_heads * d_head, d_model)
    for i in range(num_heads):
        start_idx = i * d_head
        end_idx = (i + 1) * d_head
        attn_weights[f'k_heads.{i}.weight'] = k_proj_weight[start_idx:end_idx, :]
        
    # Split consolidated attn.v_proj.weight into individual v_heads.{i}.weight
    v_proj_weight = weights['attn.v_proj.weight']  # Shape: (num_heads * d_head, d_model) 
    for i in range(num_heads):
        start_idx = i * d_head
        end_idx = (i + 1) * d_head
        attn_weights[f'v_heads.{i}.weight'] = v_proj_weight[start_idx:end_idx, :]
        
    # Copy output projection weight
    attn_weights['output_proj.weight'] = weights['attn.output_proj.weight']
    
    multihead_self_attention_output = multihead_self_attention_implemented(d_model, num_heads, attn_pdrop, attn_weights, rmsnorm1_output)
    
    # Step 3: Apply dropout and residual connection
    dropout_multihead_output = torch.nn.Dropout(residual_pdrop)(multihead_self_attention_output)
    att_residual = in_features + dropout_multihead_output

    # SECOND SUBLAYER: FFN with pre-norm
    # Step 4: Apply second RMSNorm  
    ln2_weights = {'weight': weights['ln2.weight']}
    rmsnorm2 = RMSNormImplemented(d_model, ln2_weights, eps=1e-5)
    rmsnorm2_output = rmsnorm2(att_residual)
    
    # Step 5: Apply FFN
    ffn_weights = {'w1.weight': weights['ffn.w1.weight'], 'w2.weight': weights['ffn.w2.weight']}
    ffn = PositionwiseFeedforwardImplemented(d_model, d_ff, ffn_weights)
    feedforward_output = ffn(rmsnorm2_output)
    
    # Step 6: Apply dropout and final residual connection
    dropout_ffn_output = torch.nn.Dropout(residual_pdrop)(feedforward_output)
    
    return att_residual + dropout_ffn_output


def transformer_lm_implemented(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `position_embeddings.weight`
                Positional embedding matrix. Shape is (context_length, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices: torch.LongTensor
            Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the predicted unnormalized
        next-word distribution for each token.
    """
    token_embedding_layer = torch.nn.Embedding(vocab_size, d_model)
    token_embedding_layer.weight.data = weights['token_embeddings.weight']

    position_embedding_layer = torch.nn.Embedding(context_length, d_model)
    position_embedding_layer.weight.data = weights['position_embeddings.weight']

    
    ln_final_weights = {'weight': weights['ln_final.weight']}
    ln_final = RMSNormImplemented(d_model, ln_final_weights, eps=1e-5)
    lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
    lm_head.weight.data = weights['lm_head.weight']
    
    # in_indices is (batch_size, sequence_length)
    # x_after_token_embeddings, x_after_position_embeddings is (batch_size, sequence_length, d_model)
    # x_after_token_and_position_embeddings is (batch_size, sequence_length, d_model)
    batch_size, sequence_length = in_indices.shape
    position_indices = torch.arange(sequence_length, device=in_indices.device)
    position_indices = position_indices.unsqueeze(0).expand(batch_size, sequence_length)

    x_after_token_embeddings = token_embedding_layer(in_indices)
    x_after_position_embeddings = position_embedding_layer(position_indices)  # (batch_size, seq_len, d_model)
    x_after_token_and_position_embeddings = x_after_token_embeddings + x_after_position_embeddings
    dropout_layer = torch.nn.Dropout(residual_pdrop)
    # dropout_layer.eval()  # Disable dropout
    x_after_token_and_position_embeddings = dropout_layer(x_after_token_and_position_embeddings)

    for i in range(num_layers):
        weights_layer_i = {
            'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
            'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
            'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
            'attn.output_proj.weight': weights[f'layers.{i}.attn.output_proj.weight'],
            'ln1.weight': weights[f'layers.{i}.ln1.weight'],
            'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
            'ffn.w2.weight': weights[f'layers.{i}.ffn.w2.weight'],
            'ln2.weight': weights[f'layers.{i}.ln2.weight']
        }
        x_after_token_and_position_embeddings = transformer_block_implemented(
            d_model, 
            num_heads, 
            d_ff, 
            attn_pdrop, 
            residual_pdrop, 
            weights_layer_i, 
            x_after_token_and_position_embeddings
        )
    
    x_after_transformer_blocks = ln_final(x_after_token_and_position_embeddings)
    output = lm_head(x_after_transformer_blocks)
    return output

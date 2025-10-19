# Complete Training Pipeline

This guide walks you through the complete pipeline from raw text to a trained model.

## Overview

The training pipeline consists of three main steps:

1. **Train a BPE tokenizer** (if you don't have one already)
2. **Prepare your data** - Convert text to token IDs
3. **Train your model** - Run the training loop

## Step-by-Step Guide

### Step 1: Train a BPE Tokenizer

First, train a BPE tokenizer on your dataset:

```python
from cs336_bpe.bpe_encoder import run_train_bpe

# Train tokenizer
vocab, merges = run_train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=4096,
    special_tokens=["<|endoftext|>"],
)

# Save vocab and merges
import json

with open("vocab.json", "w") as f:
    json.dump({k: list(v) for k, v in vocab.items()}, f)

with open("merges.txt", "w") as f:
    for merge in merges:
        f.write(f"{merge[0].decode('utf-8', errors='replace')} {merge[1].decode('utf-8', errors='replace')}\n")
```

### Step 2: Prepare Your Data

Convert your text files to memory-mapped numpy arrays:

```bash
# Prepare training data
python prepare_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories_train.npy \
  --vocab vocab.json \
  --merges merges.txt

# Prepare validation data
python prepare_data.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/tinystories_val.npy \
  --vocab vocab.json \
  --merges merges.txt
```

This will:
- Load your BPE tokenizer
- Encode the text to token IDs
- Save as a `.npy` file that can be memory-mapped for efficient training

### Step 3: Train Your Model

Now train your Transformer model:

```bash
python train_example.py \
  --train_data data/tinystories_train.npy \
  --val_data data/tinystories_val.npy \
  --vocab_size 4096 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 8 \
  --num_heads 8 \
  --d_ff 2048 \
  --batch_size 64 \
  --max_iters 50000 \
  --max_lr 6e-4 \
  --min_lr 6e-5 \
  --warmup_iters 500 \
  --cosine_cycle_iters 50000 \
  --eval_interval 500 \
  --checkpoint_interval 5000 \
  --checkpoint_dir ./checkpoints/tinystories \
  --device cuda
```

### Step 4: Monitor Training

The training script will output progress to console:

```
================================================================================
Training Configuration:
--------------------------------------------------------------------------------
Model: 8L x 512D x 8H
Training data: data/tinystories_train.npy
Validation data: data/tinystories_val.npy
Batch size: 64
Context length: 256
Max iterations: 50000
Device: cuda
Checkpoint dir: ./checkpoints/tinystories
================================================================================
Loading datasets...
Training dataset size: 30,000,000 tokens
Validation dataset size: 2,000,000 tokens
Creating model...
Model parameters: 42,598,400
Creating optimizer...
Starting training...
iter      0 | train loss 10.8234 | lr 1.20e-05 | time 245.32ms
iter     10 | train loss 9.2341 | lr 1.32e-04 | time 142.18ms
...
iter    500 | train loss 3.2341 | val loss 3.5432
Saved checkpoint to ./checkpoints/tinystories/checkpoint_iter_5000.pt
...
```

### Step 5: Resume Training (if needed)

If training is interrupted, you can resume from a checkpoint:

```bash
python train_example.py \
  --train_data data/tinystories_train.npy \
  --val_data data/tinystories_val.npy \
  --resume_from ./checkpoints/tinystories/checkpoint_iter_5000.pt \
  ... # other args
```

## Architecture Overview

### Training Loop Features

The `cs336_torch/training_loop.py` module provides:

```python
@dataclass
class TrainingConfig:
    """All hyperparameters in one place"""
    # Model architecture
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    
    # Optimization
    learning_rate: float
    weight_decay: float
    grad_clip: float
    
    # Training
    batch_size: int
    max_iters: int
    eval_interval: int
    checkpoint_interval: int
    ...

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_dataset: np.ndarray,  # Memory-mapped!
    val_dataset: Optional[np.ndarray],
    config: TrainingConfig,
    lr_schedule_fn: Optional[Callable],  # Cosine schedule with warmup
    grad_clip_fn: Optional[Callable],    # Gradient clipping
    logger: Optional[Callable],          # External logging (W&B, etc.)
    resume_from: Optional[str],          # Resume from checkpoint
):
    """Main training loop with all the bells and whistles"""
    ...
```

### Data Loading

The `get_batch_implemented` function in `cs336_torch/data_loader.py` samples random contiguous sequences:

```python
def get_batch_implemented(
    dataset: np.ndarray,  # Can be memory-mapped!
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x: Input sequences (batch_size, context_length)
        y: Target sequences (batch_size, context_length)
           where y[i, j] = x[i, j+1] (next token prediction)
    """
```

### Checkpointing

The `cs336_torch/checkpoint.py` module provides:

```python
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | Path,
):
    """Save model, optimizer, and iteration number"""

def load_checkpoint(
    src: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint and return iteration number"""
```

## Advanced Usage

### Using Weights & Biases

Track your experiments with W&B:

```bash
# Install wandb
pip install wandb

# Login (first time only)
wandb login

# Train with W&B logging
python train_example.py \
  --train_data data/tinystories_train.npy \
  --use_wandb \
  --wandb_project my-transformer-experiments \
  ... # other args
```

### Custom Model Integration

To use your own Transformer model, modify `train_example.py`:

```python
def create_model(config: TrainingConfig) -> torch.nn.Module:
    from cs336_torch.transformer_lm import TransformerLM  # Your implementation
    
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
    )
    return model
```

### Custom Loss Function

To use your own cross-entropy implementation:

```python
def create_loss_fn() -> torch.nn.Module:
    from cs336_torch.cross_entropy import CrossEntropyLoss  # Your implementation
    return CrossEntropyLoss()
```

### Hyperparameter Sweeps

For hyperparameter tuning, you can use the config dataclass:

```python
from cs336_torch.training_loop import TrainingConfig, train

configs_to_try = [
    TrainingConfig(
        vocab_size=4096,
        context_length=256,
        d_model=d_model,
        num_layers=num_layers,
        ...
    )
    for d_model in [256, 512, 768]
    for num_layers in [4, 6, 8]
]

for i, config in enumerate(configs_to_try):
    config.checkpoint_dir = f"./checkpoints/sweep_{i}"
    # Create and train model with this config
    ...
```

## Memory Requirements

### Model Size

Approximate parameters for different configurations:

| Config | Parameters | GPU Memory (FP32) |
|--------|-----------|------------------|
| Tiny (4L, 256D) | ~10M | ~1GB |
| Small (6L, 512D) | ~40M | ~2GB |
| Medium (12L, 768D) | ~125M | ~4GB |
| Large (24L, 1024D) | ~350M | ~8GB |

Add batch_size * context_length * d_model * 4 bytes for activations.

### Dataset Size

Memory-mapped datasets don't consume RAM, only disk space:

- TinyStories: ~300M tokens × 4 bytes = ~1.2GB on disk
- OpenWebText (sample): ~1B tokens × 4 bytes = ~4GB on disk

## Troubleshooting

See [TRAINING.md](TRAINING.md) for detailed troubleshooting guide.

### Common Issues

**Out of Memory**: Reduce batch size or model size
**Slow Training**: Use GPU, increase batch size
**NaN Loss**: Reduce learning rate, increase gradient clipping
**No Improvement**: Check data preprocessing, increase learning rate

## File Structure

After following this guide, you should have:

```
.
├── data/
│   ├── TinyStoriesV2-GPT4-train.txt    # Raw text
│   ├── TinyStoriesV2-GPT4-valid.txt
│   ├── tinystories_train.npy           # Encoded tokens
│   └── tinystories_val.npy
├── vocab.json                           # BPE vocabulary
├── merges.txt                           # BPE merges
├── checkpoints/
│   └── tinystories/
│       ├── checkpoint_iter_5000.pt
│       ├── checkpoint_iter_10000.pt
│       └── ...
├── train_example.py                     # Training script
├── prepare_data.py                      # Data preparation script
└── cs336_torch/
    ├── training_loop.py                 # Training loop implementation
    ├── data_loader.py                   # Data loading
    ├── checkpoint.py                    # Checkpoint management
    ├── adamw.py                         # AdamW optimizer
    ├── gradient_clipping.py             # Gradient clipping
    ├── lr_cosine_schedule.py           # LR schedule
    └── ...                              # Your model implementations
```

## Next Steps

1. Implement your Transformer model components
2. Train on small dataset to verify everything works
3. Scale up to full dataset
4. Tune hyperparameters
5. Evaluate model performance
6. Generate text samples

For more details, see:
- [TRAINING.md](TRAINING.md) - Complete training documentation
- [README.md](README.md) - Assignment overview and setup


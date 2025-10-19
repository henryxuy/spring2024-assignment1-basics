# Training Script Documentation

This document explains how to use the training loop implementation to train Transformer language models.

## Features

The training loop implementation (`cs336_torch/training_loop.py`) provides:

1. **Hyperparameter Configuration**: Control all model and optimizer hyperparameters via `TrainingConfig` dataclass
2. **Memory-Efficient Data Loading**: Use `np.memmap` for large datasets without loading everything into RAM
3. **Checkpoint Management**: Automatic checkpoint saving at configurable intervals with resume support
4. **Training & Validation Logging**: Periodic logging of training/validation metrics to console and external services (e.g., Weights & Biases)
5. **Learning Rate Scheduling**: Cosine decay with linear warmup
6. **Gradient Clipping**: Configurable gradient norm clipping

## Quick Start

### 1. Prepare Your Data

Convert your tokenized text data to a numpy array of token IDs and save as `.npy`:

```python
import numpy as np

# Assuming you have a list/array of token IDs
token_ids = [1, 2, 3, ...]  # Your tokenized data
np.save("train_data.npy", np.array(token_ids, dtype=np.int32))
```

### 2. Basic Training Command

```bash
python train_example.py \
  --train_data data/train_data.npy \
  --val_data data/val_data.npy \
  --vocab_size 50257 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 6 \
  --num_heads 8 \
  --d_ff 2048 \
  --batch_size 32 \
  --max_iters 10000 \
  --checkpoint_dir ./checkpoints
```

### 3. Resume from Checkpoint

```bash
python train_example.py \
  --train_data data/train_data.npy \
  --resume_from checkpoints/checkpoint_iter_5000.pt \
  ... # other args
```

### 4. Use Weights & Biases Logging

```bash
# Install wandb first: pip install wandb
python train_example.py \
  --train_data data/train_data.npy \
  --use_wandb \
  --wandb_project my-transformer-project \
  ... # other args
```

## Configuration Options

### Model Hyperparameters

- `--vocab_size`: Vocabulary size (default: 50257)
- `--context_length`: Maximum sequence length (default: 256)
- `--d_model`: Model dimension (default: 512)
- `--num_layers`: Number of transformer layers (default: 6)
- `--num_heads`: Number of attention heads (default: 8)
- `--d_ff`: Feedforward network dimension (default: 2048)
- `--attn_pdrop`: Attention dropout probability (default: 0.1)
- `--residual_pdrop`: Residual dropout probability (default: 0.1)

### Optimizer Hyperparameters

- `--learning_rate`: Initial learning rate (default: 3e-4)
- `--weight_decay`: Weight decay (L2 regularization) (default: 0.1)
- `--beta1`: Adam beta1 (default: 0.9)
- `--beta2`: Adam beta2 (default: 0.95)

### Learning Rate Schedule

- `--max_lr`: Maximum learning rate (default: 3e-4)
- `--min_lr`: Minimum learning rate (default: 3e-5)
- `--warmup_iters`: Number of warmup iterations (default: 100)
- `--cosine_cycle_iters`: Total iterations for cosine decay (default: 10000)

### Training Configuration

- `--batch_size`: Batch size (default: 32)
- `--max_iters`: Maximum training iterations (default: 10000)
- `--eval_interval`: Evaluate every N iterations (default: 100)
- `--eval_iters`: Number of batches for evaluation (default: 10)
- `--log_interval`: Log metrics every N iterations (default: 10)
- `--checkpoint_interval`: Save checkpoint every N iterations (default: 1000)
- `--grad_clip`: Gradient clipping threshold (default: 1.0, 0 to disable)

### Paths and Logging

- `--checkpoint_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--resume_from`: Path to checkpoint to resume from
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name (default: transformer-lm)

### Device

- `--device`: Device to use (cuda/cpu, default: auto-detect)

## Using the Training Loop Programmatically

You can also use the training loop directly in your Python code:

```python
import numpy as np
import torch
from cs336_torch.training_loop import train, TrainingConfig, load_dataset_memmap
from cs336_torch.adamw import AdamWImplemented

# Create configuration
config = TrainingConfig(
    vocab_size=50257,
    context_length=256,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    batch_size=32,
    max_iters=10000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="./checkpoints",
)

# Load datasets with memory mapping
train_dataset = load_dataset_memmap("train_data.npy")
val_dataset = load_dataset_memmap("val_data.npy")

# Create your model
model = YourTransformerModel(...)

# Create optimizer
optimizer = AdamWImplemented(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2),
)

# Create loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Optional: Create custom logger
def my_logger(metrics):
    print(f"Custom log: {metrics}")

# Train
train(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config,
    logger=my_logger,
)
```

## Memory-Efficient Data Loading

The training loop supports memory-mapped datasets to handle large files efficiently:

```python
from cs336_torch.training_loop import load_dataset_memmap

# For .npy files (recommended)
dataset = load_dataset_memmap("data.npy")

# For raw binary files
dataset = load_dataset_memmap("data.bin")
```

Memory mapping loads data on-demand rather than loading the entire dataset into RAM, making it possible to train on datasets larger than your available memory.

## Checkpoint Format

Checkpoints are saved as PyTorch state dictionaries containing:

```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'iteration': iteration_number,
}
```

You can load and inspect checkpoints:

```python
import torch

checkpoint = torch.load("checkpoints/checkpoint_iter_5000.pt")
print(f"Checkpoint at iteration: {checkpoint['iteration']}")
```

## Example: TinyStories Training

Here's a complete example for training on the TinyStories dataset:

```bash
# 1. Tokenize your data (you'll need to implement this based on your tokenizer)
python tokenize_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories_train.npy \
  --vocab vocab.json \
  --merges merges.txt

# 2. Train the model
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

## Monitoring Training

### Console Output

The training loop prints progress to console:

```
iter      0 | train loss 10.8234 | lr 6.00e-06 | time 245.32ms
iter     10 | train loss 9.2341 | lr 6.60e-05 | time 142.18ms
iter    100 | train loss 5.2341 | val loss 5.5432
...
Saved checkpoint to ./checkpoints/checkpoint_iter_1000.pt
```

### Weights & Biases

When using `--use_wandb`, metrics are automatically logged to W&B including:

- `train_loss`: Training loss per iteration
- `train_loss_avg`: Average training loss over multiple batches
- `val_loss`: Validation loss (if validation data provided)
- `learning_rate`: Current learning rate
- `iteration`: Current iteration

## Best Practices

1. **Start Small**: Test your setup with a small model and dataset first
2. **Monitor Validation Loss**: Always use a validation set to detect overfitting
3. **Checkpoint Frequently**: Balance disk space with the cost of re-training
4. **Use Memory Mapping**: For datasets > 1GB, always use `.npy` files with memory mapping
5. **Tune Learning Rate**: The learning rate is often the most important hyperparameter
6. **Watch for NaN**: If you see NaN losses, try reducing learning rate or increasing gradient clipping

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Reduce `--context_length`
- Reduce model size (`--d_model`, `--num_layers`, `--d_ff`)
- Ensure you're using memory-mapped data loading

### Slow Training

- Increase `--batch_size` if memory allows
- Use GPU (`--device cuda`)
- Reduce `--eval_interval` and `--log_interval`

### Loss Not Decreasing

- Increase `--max_lr` (try 1e-3 to 6e-4)
- Check your data preprocessing
- Verify model implementation
- Try longer warmup (`--warmup_iters`)

### Checkpoint Not Saving

- Verify `--checkpoint_dir` exists and is writable
- Check disk space
- Ensure checkpoint interval is reached


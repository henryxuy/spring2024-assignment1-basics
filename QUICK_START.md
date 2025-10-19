# Quick Start Guide

This is a quick reference for training your Transformer language model.

## Three-Step Training Process

### 1️⃣ Prepare Data

```bash
python prepare_data.py \
  --input data/train.txt \
  --output data/train.npy \
  --vocab vocab.json \
  --merges merges.txt
```

### 2️⃣ Train Model

```bash
python train_example.py \
  --train_data data/train.npy \
  --val_data data/val.npy \
  --vocab_size 4096 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 8 \
  --batch_size 64 \
  --max_iters 50000 \
  --checkpoint_dir ./checkpoints
```

### 3️⃣ Resume if Interrupted

```bash
python train_example.py \
  --resume_from ./checkpoints/checkpoint_iter_10000.pt \
  ... # same args as before
```

## Essential Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train_data` | Training data (.npy) | Required |
| `--val_data` | Validation data (.npy) | None |
| `--vocab_size` | Vocabulary size | 50257 |
| `--context_length` | Sequence length | 256 |
| `--d_model` | Model dimension | 512 |
| `--num_layers` | Number of layers | 6 |
| `--num_heads` | Attention heads | 8 |
| `--batch_size` | Batch size | 32 |
| `--max_iters` | Training iterations | 10000 |
| `--checkpoint_dir` | Checkpoint directory | ./checkpoints |
| `--device` | Device (cuda/cpu) | Auto |

## Common Configurations

### Tiny (for testing)
```bash
--d_model 256 --num_layers 4 --num_heads 4 --d_ff 1024 --batch_size 32
```

### Small
```bash
--d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 --batch_size 64
```

### Medium
```bash
--d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072 --batch_size 32
```

## Key Features

✅ **Hyperparameter Control** - Configure everything via command-line args  
✅ **Memory-Efficient** - Use np.memmap for large datasets  
✅ **Checkpointing** - Auto-save and resume from checkpoints  
✅ **Logging** - Console output + optional Weights & Biases  
✅ **LR Schedule** - Cosine decay with warmup  
✅ **Gradient Clipping** - Stable training with configurable clipping  

## Detailed Documentation

- 📖 [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - Complete end-to-end guide
- 📖 [TRAINING.md](TRAINING.md) - Detailed training documentation
- 📖 [README.md](README.md) - Assignment overview

## Help

```bash
python train_example.py --help
python prepare_data.py --help
```


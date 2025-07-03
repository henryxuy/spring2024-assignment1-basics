# Product Requirements Document: Transformer Language Model Training

## Overview
Build and train a Transformer language model from scratch using PyTorch, implementing all core components without using high-level abstractions.

## Components to Implement

### 1. Tokenizer
- **Byte-pair encoding (BPE) tokenizer**
- Train on TinyStories dataset
- Convert text to integer ID sequences

### 2. Model Architecture
- **Transformer language model (LM)**
- Built from scratch using allowed PyTorch components only

### 3. Training Infrastructure
- **Cross-entropy loss function**
- **AdamW optimizer**
- **Training loop** with state serialization and loading support

## Tasks to Execute

### 1. Tokenizer Training
- Train BPE tokenizer on TinyStories dataset
- Apply trained tokenizer to convert dataset to integer sequences

### 2. Model Training & Evaluation
- Train Transformer LM on TinyStories dataset
- Generate text samples from trained model
- Evaluate model perplexity

### 3. Leaderboard Submission
- Train models on OpenWebText dataset
- Submit perplexity results to leaderboard

## Technical Constraints

### Allowed PyTorch Components
- `torch.nn.Parameter`
- `torch.nn.Dropout` and `torch.nn.functional.dropout`
- `torch.nn.Linear` and `torch.nn.Embedding`
- Container classes: `torch.nn.Module`, `torch.nn.ModuleList`, `torch.nn.Sequential`
- `torch.optim.Optimizer` base class
- Any other PyTorch definitions not in `torch.nn`, `torch.nn.functional`, or `torch.optim`

### Prohibited
- Pre-built components from `torch.nn`, `torch.nn.functional`, `torch.optim` (except those listed above)

## Success Criteria
- All components implemented from scratch
- Successful training on TinyStories dataset
- Text generation capability
- Perplexity evaluation working
- OpenWebText training results submitted

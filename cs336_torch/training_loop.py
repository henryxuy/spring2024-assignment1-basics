import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import os
from pathlib import Path
import time

from .checkpoint import save_checkpoint, load_checkpoint
from .data_loader import get_batch_implemented


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    # Model hyperparameters
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    attn_pdrop: float = 0.1
    residual_pdrop: float = 0.1
    
    # Optimizer hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Learning rate schedule
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    cosine_cycle_iters: int = 10000
    
    # Training hyperparameters
    batch_size: int = 32
    max_iters: int = 10000
    eval_interval: int = 100
    eval_iters: int = 10
    log_interval: int = 10
    checkpoint_interval: int = 1000
    
    # Gradient clipping
    grad_clip: Optional[float] = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None


def load_dataset_memmap(data_path: str) -> np.ndarray:
    """
    Load a dataset using memory mapping for efficient memory usage.
    
    Args:
        data_path: Path to the numpy array file (.npy) containing token IDs.
    
    Returns:
        Memory-mapped numpy array of token IDs.
    """
    if data_path.endswith('.npy'):
        # Use memory mapping for .npy files
        dataset = np.load(data_path, mmap_mode='r')
    elif data_path.endswith('.bin'):
        # For raw binary files, create memmap directly
        dataset = np.memmap(data_path, dtype=np.int32, mode='r')
    else:
        # Fallback: load into memory
        dataset = np.load(data_path)
    
    return dataset


def get_lr(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Get learning rate for current iteration using cosine schedule with warmup.
    
    Args:
        iteration: Current training iteration.
        max_learning_rate: Maximum learning rate after warmup.
        min_learning_rate: Minimum learning rate at end of cosine cycle.
        warmup_iters: Number of warmup iterations.
        cosine_cycle_iters: Total iterations for cosine decay cycle.
    
    Returns:
        Learning rate for current iteration.
    """
    # Linear warmup
    if iteration < warmup_iters:
        return max_learning_rate * (iteration + 1) / warmup_iters
    
    # Cosine decay
    if iteration < cosine_cycle_iters:
        decay_ratio = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
    
    # After cosine cycle, use minimum learning rate
    return min_learning_rate


def estimate_loss(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataset: np.ndarray,
    config: TrainingConfig,
    num_iters: int,
) -> float:
    """
    Estimate loss over multiple batches.
    
    Args:
        model: The model to evaluate.
        loss_fn: Loss function.
        dataset: Dataset to sample from.
        config: Training configuration.
        num_iters: Number of batches to average over.
    
    Returns:
        Average loss.
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for _ in range(num_iters):
            x, y = get_batch_implemented(
                dataset=dataset,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=config.device,
            )
            logits = model(x)
            # Reshape for cross-entropy: (batch_size * context_length, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            targets = y.view(-1)
            loss = loss_fn(logits, targets)
            losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_dataset: np.ndarray,
    val_dataset: Optional[np.ndarray],
    config: TrainingConfig,
    lr_schedule_fn: Optional[Callable[[int], float]] = None,
    grad_clip_fn: Optional[Callable[[torch.nn.Module, float], None]] = None,
    logger: Optional[Callable[[dict], None]] = None,
    resume_from: Optional[str] = None,
) -> None:
    """
    Main training loop with checkpointing, logging, and evaluation.
    
    Args:
        model: The model to train.
        optimizer: Optimizer for training.
        loss_fn: Loss function.
        train_dataset: Training dataset (memory-mapped numpy array).
        val_dataset: Validation dataset (memory-mapped numpy array), optional.
        config: Training configuration.
        lr_schedule_fn: Optional function to compute learning rate given iteration.
        grad_clip_fn: Optional function to clip gradients.
        logger: Optional logging function that takes a dict of metrics.
        resume_from: Optional path to checkpoint to resume from.
    """
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint if provided
    start_iter = 0
    if resume_from and os.path.exists(resume_from):
        start_iter = load_checkpoint(resume_from, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")
    
    model.to(config.device)
    model.train()
    
    # Training loop
    t0 = time.time()
    for iteration in range(start_iter, config.max_iters):
        # Update learning rate
        if lr_schedule_fn is not None:
            lr = lr_schedule_fn(iteration)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer.param_groups[0]['lr']
        
        # Get batch
        x, y = get_batch_implemented(
            dataset=train_dataset,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        
        # Forward pass
        logits = model(x)
        
        # Reshape for cross-entropy: (batch_size * context_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        targets = y.view(-1)
        loss = loss_fn(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip_fn is not None and config.grad_clip is not None:
            grad_clip_fn(model, config.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iteration % config.log_interval == 0 or iteration == config.max_iters - 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            metrics = {
                'iteration': iteration,
                'train_loss': loss.item(),
                'learning_rate': lr,
                'time_ms': dt * 1000,
            }
            
            # Log to console
            print(f"iter {iteration:6d} | train loss {loss.item():.4f} | lr {lr:.2e} | time {dt*1000:.2f}ms")
            
            # Call external logger if provided
            if logger is not None:
                logger(metrics)
        
        # Evaluation
        if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
            # Evaluate on training set
            train_loss = estimate_loss(model, loss_fn, train_dataset, config, config.eval_iters)
            
            metrics = {
                'iteration': iteration,
                'train_loss_avg': train_loss,
            }
            
            # Evaluate on validation set if provided
            if val_dataset is not None:
                val_loss = estimate_loss(model, loss_fn, val_dataset, config, config.eval_iters)
                metrics['val_loss'] = val_loss
                print(f"iter {iteration:6d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
            else:
                print(f"iter {iteration:6d} | train loss {train_loss:.4f}")
            
            # Call external logger if provided
            if logger is not None:
                logger(metrics)
        
        # Checkpoint saving
        if iteration % config.checkpoint_interval == 0 or iteration == config.max_iters - 1:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training complete!")


def sample_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> None:
    """
    Simple training loop example (for backward compatibility).
    
    Note: For production use, prefer the `train()` function which supports
    checkpointing, validation, logging, and memory-mapped datasets.
    """
    # Train the model
    for batch in data_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            print(loss.item())
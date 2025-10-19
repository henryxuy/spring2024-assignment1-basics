#!/usr/bin/env python3
"""
Example training script demonstrating how to use the training loop.

This script shows how to:
- Configure model and optimizer hyperparameters
- Load datasets with memory mapping
- Train with checkpointing
- Log metrics to console and optionally to Weights & Biases
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from cs336_torch.training_loop import train, TrainingConfig, load_dataset_memmap, get_lr
from cs336_torch.checkpoint import load_checkpoint
from cs336_torch.adamw import AdamWImplemented
from cs336_torch.gradient_clipping import clip_grad_norm_


def create_model(config: TrainingConfig) -> torch.nn.Module:
    """
    Create a Transformer language model.
    
    Note: Replace this with your actual model implementation.
    """
    # Placeholder - replace with your actual Transformer model
    # from cs336_torch.transformer_lm import TransformerLM
    # model = TransformerLM(
    #     vocab_size=config.vocab_size,
    #     context_length=config.context_length,
    #     d_model=config.d_model,
    #     num_layers=config.num_layers,
    #     num_heads=config.num_heads,
    #     d_ff=config.d_ff,
    #     attn_pdrop=config.attn_pdrop,
    #     residual_pdrop=config.residual_pdrop,
    # )
    
    # For demonstration, create a simple model
    model = torch.nn.Sequential(
        torch.nn.Embedding(config.vocab_size, config.d_model),
        torch.nn.Linear(config.d_model, config.vocab_size)
    )
    return model


def create_loss_fn() -> torch.nn.Module:
    """
    Create cross-entropy loss function.
    
    Note: Replace with your custom implementation if needed.
    """
    # from cs336_torch.cross_entropy import CrossEntropyLoss
    # return CrossEntropyLoss()
    
    # For demonstration, use PyTorch's built-in
    return torch.nn.CrossEntropyLoss()


def create_logger(use_wandb: bool = False, project_name: str = "transformer-lm"):
    """
    Create a logger function for training metrics.
    
    Args:
        use_wandb: Whether to use Weights & Biases for logging.
        project_name: W&B project name.
    
    Returns:
        Logger function that takes a dict of metrics.
    """
    if use_wandb:
        try:
            import wandb
            wandb.init(project=project_name)
            
            def logger(metrics: dict):
                wandb.log(metrics)
            
            return logger
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            return None
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    
    # Data paths
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (.npy file with token IDs)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data (.npy file with token IDs)")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Context length")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feedforward dimension")
    parser.add_argument("--attn_pdrop", type=float, default=0.1,
                        help="Attention dropout probability")
    parser.add_argument("--residual_pdrop", type=float, default=0.1,
                        help="Residual dropout probability")
    
    # Optimizer hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Adam beta2")
    
    # Learning rate schedule
    parser.add_argument("--max_lr", type=float, default=3e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=100,
                        help="Number of warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000,
                        help="Total iterations for cosine cycle")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_iters", type=int, default=10000,
                        help="Maximum number of training iterations")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=10,
                        help="Number of iterations for evaluation")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Checkpoint saving interval")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold (None to disable)")
    
    # Paths and logging
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm",
                        help="W&B project name")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir=args.checkpoint_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    
    print("=" * 80)
    print("Training Configuration:")
    print("-" * 80)
    print(f"Model: {config.num_layers}L x {config.d_model}D x {config.num_heads}H")
    print(f"Training data: {config.train_data_path}")
    print(f"Validation data: {config.val_data_path}")
    print(f"Batch size: {config.batch_size}")
    print(f"Context length: {config.context_length}")
    print(f"Max iterations: {config.max_iters}")
    print(f"Device: {config.device}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print("=" * 80)
    
    # Load datasets with memory mapping
    print("Loading datasets...")
    train_dataset = load_dataset_memmap(config.train_data_path)
    print(f"Training dataset size: {len(train_dataset):,} tokens")
    
    val_dataset = None
    if config.val_data_path:
        val_dataset = load_dataset_memmap(config.val_data_path)
        print(f"Validation dataset size: {len(val_dataset):,} tokens")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = AdamWImplemented(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    
    # Create loss function
    loss_fn = create_loss_fn()
    
    # Create learning rate schedule function
    def lr_schedule_fn(iteration: int) -> float:
        return get_lr(
            iteration,
            config.max_learning_rate,
            config.min_learning_rate,
            config.warmup_iters,
            config.cosine_cycle_iters,
        )
    
    # Create gradient clipping function
    def grad_clip_fn(model: torch.nn.Module, max_norm: float):
        clip_grad_norm_(model.parameters(), max_norm)
    
    # Create logger
    logger = create_logger(args.use_wandb, args.wandb_project)
    
    # Train
    print("Starting training...")
    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        lr_schedule_fn=lr_schedule_fn,
        grad_clip_fn=grad_clip_fn if config.grad_clip else None,
        logger=logger,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()


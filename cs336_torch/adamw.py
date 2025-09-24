import torch
from collections import defaultdict
from typing import Optional, Callable
import math

class AdamWImplemented(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):   
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} (must be non-negative)")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]} (must be in [0.0, 1.0))")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]} (must be in [0.0, 1.0))")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay} (must be non-negative)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} (must be non-negative)")
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        
        # Call parent class initialization - this automatically handles:
        super().__init__(params, defaults)
        
        # What super().__init__(params, defaults) does automatically:
        # 
        # 1. Initialize self.state:
        #    self.state = defaultdict(dict)
        #    - Creates empty state dictionary for storing per-parameter optimization state
        #    - Each parameter will get its own state dict (e.g., momentum buffers, step counts)
        #
        # 2. Initialize self.param_groups:
        #    - Processes the input 'params' (handles single tensors, lists, generators)
        #    - Validates that parameters are not empty
        #    - Creates structured parameter groups by merging with defaults:
        #    self.param_groups = [
        #        {
        #            'params': [list_of_actual_parameter_tensors],  # Processed parameter list
        #            'lr': defaults['lr'],                          # Learning rate
        #            'weight_decay': defaults['weight_decay'],      # Weight decay coefficient  
        #            'betas': defaults['betas'],                    # Momentum coefficients (β₁, β₂)
        #            'eps': defaults['eps']                         # Numerical stability constant
        #        }
        #    ]
        #
        # 3. Additional parent class setup:
        #    - Handles edge cases (empty parameter lists, invalid inputs)
        #    - Ensures parameters are properly formatted for optimization
        #    - Sets up infrastructure for methods like zero_grad(), state_dict(), etc.
        #
        # Result: self.state and self.param_groups are ready to use - no manual initialization needed!
        

    def step(self, closure: Optional[Callable] = None):
        # one step of AdamW
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            betas = group['betas']
            eps = group['eps']
            # p: weight tensor
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                
                # Increment step count FIRST
                t = state.get('t', 0) + 1
                state['t'] = t
                
                # Update exponential moving averages
                state['m'] = betas[0] * state.get('m', 0) + (1 - betas[0]) * grad
                state['v'] = betas[1] * state.get('v', 0) + (1 - betas[1]) * grad**2
 
                # Compute adjusted learning rate with bias correction
                # αt ← α * √(1−(β2)^t) / √(1−(β1)^t)
                lr_t = lr * math.sqrt(1 - betas[1]**t) / math.sqrt(1 - betas[0]**t)
                
                # AdamW update: Apply in two sequential steps as per algorithm
                
                # Step 1: θ ← θ − αt * m/(√v+ϵ)
                adaptive_term = state['m'] / (torch.sqrt(state['v']) + eps)
                p.data -= lr_t * adaptive_term
                
                # Step 2: θ ← θ − α*λ*θ (Apply weight decay)
                p.data -= lr * weight_decay * p.data
        return loss

import math
import torch

class LossGuide:
    """
    Absolutely Miracle Loss Guide.
    Provides advanced learning rate scheduling with Cosine Annealing, Linear Warmup, 
    and adaptive plateau detection to inject momentum restarts and dynamically steer
    the optimization landscape towards near-zero loss.
    """
    def __init__(self, optimizer, max_steps, base_lr=8e-4, warmup_steps=50):
        self.optimizer = optimizer
        self.max_steps = int(max_steps)
        self.base_lr = base_lr
        self.warmup_steps = int(warmup_steps)
        self.loss_history = []
        self.stalled_patience = 10
        self.consecutive_stalls = 0
        self.best_loss = float('inf')
        self.exponential_boost = 1.0 # Added to expand LR exponentially
        
    def _compute_lr(self, step):
        # 1. Warmup phase
        if step < self.warmup_steps:
            base_schedule_lr = self.base_lr * (step + 1) / self.warmup_steps
        else:
            # 2. Cosine annealing phase
            progress = (step - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            # Min lr is 10% of base lr
            base_schedule_lr = self.base_lr * 0.1 + self.base_lr * 0.9 * cosine_decay

        # Apply exponential boost to learn harder
        return base_schedule_lr * self.exponential_boost

    def step(self, current_step, current_loss):
        """
        Updates the optimizer learning rate and tracks loss to break plateaus.
        """
        # Save history
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 50:
            self.loss_history.pop(0)
            
        # Exponentially increase the boost if loss is stubbornly high (> 11.0)
        if current_loss > 11.0:
            # Expand exponentially by 2% every step it remains high
            self.exponential_boost *= 1.02
        elif current_loss < 8.0 and self.exponential_boost > 1.0:
            # Gradually cool down the boost once loss drops significantly
            self.exponential_boost = max(1.0, self.exponential_boost * 0.95)

        lr = self._compute_lr(current_step)
        
        # Adaptive Plateau Breaking 🚀
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.consecutive_stalls = 0
        else:
            self.consecutive_stalls += 1
            
        if self.consecutive_stalls > self.stalled_patience:
            # We are stalled! Inject a micro-restart (warm restart boost)
            print(f"📉 [Loss Guide] Detected plateau at loss {current_loss:.4f}. Injecting exponential momentum restart!", flush=True)
            self.exponential_boost *= 1.5 # Huge exponential jump
            lr = self._compute_lr(current_step)
            self.consecutive_stalls = 0 # reset stall counter
            
            # Reset momentum buffers slightly for muon optimizer
            for group in self.optimizer.param_groups:
                if group['kind'] == 'adamw':
                    for p in group['params']:
                        state = self.optimizer.state[p]
                        if 'exp_avg' in state:
                            state['exp_avg'].mul_(0.5)  # heavy momentum decay
                elif group['kind'] == 'muon':
                    for p in group['params']:
                        state = self.optimizer.state[p]
                        if "momentum_buffer" in state:
                            state["momentum_buffer"].mul_(0.5)
        
        # Cap the max lr to avoid total divergence, but allow very high spikes
        lr = min(lr, 0.05)
        
        # Apply learning rate to all groups 
        # (Muon scales by 5.0 for matrix parameters typically)
        for group in self.optimizer.param_groups:
            if group['kind'] == 'muon':
                group['lr'] = lr * 5.0
            else:
                group['lr'] = lr
                
        return lr

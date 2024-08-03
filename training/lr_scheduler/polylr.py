from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
    def reset(self, new_initial_lr: float = None, new_max_steps: int = None, new_exponent: float = None):
        if new_initial_lr is not None:
            self.initial_lr = new_initial_lr
        if new_max_steps is not None:
            self.max_steps = new_max_steps
        if new_exponent is not None:
            self.exponent = new_exponent
        self.ctr = 0 
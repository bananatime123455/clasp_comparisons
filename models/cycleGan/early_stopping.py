class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience   = patience
        self.delta      = delta
        self.best_score = None
        self.counter    = 0
        self.early_stop = False
        self.best_states = None

    def __call__(self, val_loss, states: dict):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score  = score
            self.best_states = {k: v.copy() for k, v in states.items()}
            self.counter     = 0
            return True          # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False         # no improvement

    def restore(self, models: dict):
        """models is a dict of {name: nn.Module}."""
        for name, model in models.items():
            model.load_state_dict(self.best_states[name])
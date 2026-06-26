class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        # Store a list of (model, weights) pairs so multiple models can be tracked
        self._best_weights = {}

    def __call__(self, val_loss, models: dict):
        """
        Args:
            val_loss: Current validation loss (lower is better).
            models:   Dict of {name: tf.keras.Model} to checkpoint when improved.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_weights(models)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_weights(models)
            self.counter = 0

    def _save_weights(self, models: dict):
        for name, model in models.items():
            self._best_weights[name] = model.get_weights()

    def load_best_weights(self, models: dict):
        """Restore each model to its best recorded weights."""
        for name, model in models.items():
            if name in self._best_weights:
                model.set_weights(self._best_weights[name])
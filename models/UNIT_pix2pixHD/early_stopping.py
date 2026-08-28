import copy


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self._best_weights = {}

    def __call__(self, val_loss, models: dict):
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
            self._best_weights[name] = copy.deepcopy(model.state_dict())

    def load_best_weights(self, models: dict):
        for name, model in models.items():
            if name in self._best_weights:
                model.load_state_dict(self._best_weights[name])
import torch
import torch.nn as nn
from torch.nn import ModuleDict

class CustomAggregateLoss(nn.Module):
    """
    Args:
        losses (dict): A dictionary of instantiated loss function modules.
                       Example: {'lpips': LPIPS(), 'l1': L1Loss()}
        weights (dict): A dictionary of weights for each loss.
                        Example: {'lpips': 0.8, 'l1': 0.2}
    """
    def __init__(self, losses: dict, weights: dict):
        super().__init__()

        if losses.keys() != weights.keys():
            raise ValueError("The keys in losses and weights must match.")
        self.losses = ModuleDict(losses)
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted total loss.

        Args:
            pred (torch.Tensor): The prediction from the model.
            target (torch.Tensor): The ground truth target.

        Returns:
            torch.Tensor: The final scalar loss value.
        """

        total_loss = 0.0
        for key, loss_fn in self.losses.items():
            total_loss += self.weights[key] * loss_fn(pred, target)
        return total_loss

LAMBDA = 10.0
loss_obj = nn.BCEWithLogitsLoss()  # equivalent to BinaryCrossentropy(from_logits=True)

def generator_loss(generated):
    return loss_obj(generated, torch.ones_like(generated))

def discriminator_loss(real, generated):
    real_loss = loss_obj(real, torch.ones_like(real))
    generated_loss = loss_obj(generated, torch.zeros_like(generated))
    return (real_loss + generated_loss) * 0.5

def calc_cycle_loss(real_image, cycled_image):
    return LAMBDA * torch.mean(torch.abs(real_image - cycled_image))

def identity_loss(real_image, same_image):
    return LAMBDA * 0.5 * torch.mean(torch.abs(real_image - same_image))
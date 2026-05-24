import torch
import torch.nn as nn

LAMBDA = 10.0

adversarial_loss = nn.MSELoss()   # LSGAN — more stable than BCE for CycleGAN


def generator_loss(disc_fake):
    return adversarial_loss(disc_fake, torch.ones_like(disc_fake))


def discriminator_loss(disc_real, disc_fake):
    real_loss = adversarial_loss(disc_real, torch.ones_like(disc_real))
    fake_loss = adversarial_loss(disc_fake, torch.zeros_like(disc_fake))
    return (real_loss + fake_loss) * 0.5


def calc_cycle_loss(real, cycled):
    return LAMBDA * torch.mean(torch.abs(real - cycled))


def identity_loss(real, same):
    return LAMBDA * 0.5 * torch.mean(torch.abs(real - same))
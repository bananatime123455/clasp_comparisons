import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, channels, epsilon=1e-5):
        super().__init__()
        self.scale  = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.offset = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var  = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (var + self.epsilon).sqrt() + self.offset


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            InstanceNorm(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            InstanceNorm(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UNetGenerator(nn.Module):
    """U-Net style generator with skip connections."""
    def __init__(self, in_channels=1, out_channels=1, base=64):
        super().__init__()
        # Encoder
        self.enc1 = self._down(in_channels, base,     norm=False)
        self.enc2 = self._down(base,        base * 2)
        self.enc3 = self._down(base * 2,    base * 4)
        # Bottleneck res blocks
        self.res  = nn.Sequential(*[ResBlock(base * 4) for _ in range(6)])
        # Decoder with skip connections
        self.dec3 = self._up(base * 4,       base * 2)
        self.dec2 = self._up(base * 2 * 2,   base)       # *2 for skip concat
        self.dec1 = self._up(base * 2,        base)
        self.out  = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base, out_channels, 7),
            nn.Tanh()
        )

    def _down(self, in_c, out_c, norm=True):
        layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
        if norm:
            layers.append(InstanceNorm(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _up(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
            InstanceNorm(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.res(e3)
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # note: e1 channels must match
        return self.out(d1)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""
    def __init__(self, in_channels=1, base=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base,     4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base,       base * 2,  4, stride=2, padding=1),
            InstanceNorm(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2,   base * 4,  4, stride=2, padding=1),
            InstanceNorm(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4,   base * 8,  4, stride=1, padding=1),
            InstanceNorm(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8,   1,         4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)
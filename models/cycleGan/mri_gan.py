import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Instance Normalisation  (matches TF InstanceNormalization layer)
# ---------------------------------------------------------------------------

class InstanceNorm(nn.Module):
    """
    Per-sample, per-channel normalisation.
    Scale initialised from N(0, 0.02); offset initialised to 0 –
    matching tf.random_normal_initializer(0., 0.02) / 'zeros'.
    """

    def __init__(self, channels: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.scale  = nn.Parameter(torch.empty(1, channels, 1, 1).normal_(0.0, 0.02))
        self.offset = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[2, 3], keepdim=True)
        var  = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (var + self.epsilon).sqrt() + self.offset


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _downsample(in_c: int, out_c: int, apply_norm: bool = True) -> nn.Sequential:
    conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not apply_norm)
    nn.init.normal_(conv.weight, 0.0, 0.02)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)

    layers: list[nn.Module] = [conv]
    if apply_norm:
        layers.append(InstanceNorm(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _upsample(in_c: int, out_c: int, apply_dropout: bool = False) -> nn.Sequential:
    deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
    nn.init.normal_(deconv.weight, 0.0, 0.02)

    layers: list[nn.Module] = [deconv, InstanceNorm(out_c)]
    if apply_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# UNetGenerator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # ---- Encoder -------------------------------------------------------
        self.down1 = _downsample(in_channels, 64,  apply_norm=False)   # → 128×128×64
        self.down2 = _downsample(64,          128, apply_norm=True)    # →  64× 64×128
        self.down3 = _downsample(128,         256, apply_norm=True)    # →  32× 32×256

        # ---- Decoder (skip connections from down2, down1) ------------------
        self.up1 = _upsample(256,       128)   # → 64×64×128  (no skip yet)
        self.up2 = _upsample(128 + 128, 64)    # concat down2 → 128×128×64
        #  after up2 we concat down1 (64 ch) → 64+64 = 128 ch in final layer

        # ---- Output layer --------------------------------------------------
        last = nn.ConvTranspose2d(64 + 64, out_channels,
                                  kernel_size=4, stride=2, padding=1)
        nn.init.normal_(last.weight, 0.0, 0.02)
        self.last = nn.Sequential(last, nn.Tanh())

    def forward(self, x: torch.Tensor):
        d1  = self.down1(x)
        d2  = self.down2(d1)
        d3  = self.down3(d2)
        u1  = self.up1(d3)
        u2  = self.up2(torch.cat([u1, d2], dim=1))
        out = self.last(torch.cat([u2, d1], dim=1))
        return out, d1, d2, d3


# ---------------------------------------------------------------------------
# PatchDiscriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.down1 = _downsample(in_channels, 64,  apply_norm=False)
        self.down2 = _downsample(64,          128, apply_norm=True)
        self.down3 = _downsample(128,         256, apply_norm=True)

        # ZeroPad → Conv(256→512, stride=1) → InstanceNorm → LeakyReLU
        conv1 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0)
        nn.init.normal_(conv1.weight, 0.0, 0.02)

        conv2 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        nn.init.normal_(conv2.weight, 0.0, 0.02)

        self.head = nn.Sequential(
            nn.ZeroPad2d(1),
            conv1,
            InstanceNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d(1),
            conv2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.head(x)
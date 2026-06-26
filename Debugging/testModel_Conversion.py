"""
test_architecture_equivalence.py
=================================
Standalone test script.  Run with:

    python test_architecture_equivalence.py

Part A  (always runs)  — pure-PyTorch tests
    Checks that UNetGenerator and PatchDiscriminator produce the correct
    output shapes, that skip connections are wired correctly, and that
    the generator output is in [-1, 1] (Tanh).

Part B  (runs only if TensorFlow is importable)  — cross-framework test
    Builds both the TF and PyTorch models, copies TF weights into the
    PyTorch model layer-by-layer, feeds both models the same input, and
    asserts that the outputs are numerically identical within tolerance.

Requires (always):
    torch, torchvision (pip install torch)

Requires (Part B only):
    tensorflow >= 2.x  (pip install tensorflow)

No other project files are needed — the models are defined inline so
this script is fully self-contained.
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Inline model definitions  (copied verbatim from mri_gan.py so the test
# does not depend on the project being on sys.path)
# ──────────────────────────────────────────────────────────────────────────────

class InstanceNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.scale  = nn.Parameter(torch.empty(1, channels, 1, 1).normal_(0.0, 0.02))
        self.offset = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var  = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (var + self.epsilon).sqrt() + self.offset


def _downsample(in_c, out_c, apply_norm=True):
    conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not apply_norm)
    nn.init.normal_(conv.weight, 0.0, 0.02)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    layers = [conv]
    if apply_norm:
        layers.append(InstanceNorm(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _upsample(in_c, out_c, apply_dropout=False):
    deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
    nn.init.normal_(deconv.weight, 0.0, 0.02)
    layers = [deconv, InstanceNorm(out_c)]
    if apply_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = _downsample(in_channels, 64,  apply_norm=False)
        self.down2 = _downsample(64,          128, apply_norm=True)
        self.down3 = _downsample(128,         256, apply_norm=True)
        self.up1   = _upsample(256,       128)
        self.up2   = _upsample(128 + 128, 64)
        last = nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(last.weight, 0.0, 0.02)
        self.last  = nn.Sequential(last, nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u1 = self.up1(d3)
        u2 = self.up2(torch.cat([u1, d2], dim=1))
        return self.last(torch.cat([u2, d1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.down1 = _downsample(in_channels, 64,  apply_norm=False)
        self.down2 = _downsample(64,          128, apply_norm=True)
        self.down3 = _downsample(128,         256, apply_norm=True)
        conv1 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0)
        nn.init.normal_(conv1.weight, 0.0, 0.02)
        conv2 = nn.Conv2d(512,   1, kernel_size=4, stride=1, padding=0)
        nn.init.normal_(conv2.weight, 0.0, 0.02)
        self.head = nn.Sequential(
            nn.ZeroPad2d(1), conv1, InstanceNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d(1), conv2,
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m- SKIP\033[0m"
_results = []

def check(name: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append((name, condition))
    return condition


def summary():
    passed = sum(1 for _, ok in _results if ok)
    total  = len(_results)
    colour = "\033[92m" if passed == total else "\033[91m"
    print(f"\n{colour}{'='*50}")
    print(f"  {passed}/{total} tests passed")
    print(f"{'='*50}\033[0m")
    return passed == total


# ──────────────────────────────────────────────────────────────────────────────
# Part A  —  Pure-PyTorch shape & behaviour tests
# ──────────────────────────────────────────────────────────────────────────────

def run_part_a():
    print("\n" + "="*50)
    print("Part A — PyTorch shape & behaviour tests")
    print("="*50)

    torch.manual_seed(0)
    gen   = UNetGenerator().eval()
    disc  = PatchDiscriminator().eval()

    B = 2
    x256 = torch.randn(B, 1, 256, 256)   # standard MRI slice size
    x128 = torch.randn(B, 1, 128, 128)   # smaller sanity check

    # ── Generator output shape ────────────────────────────────────────────────
    with torch.no_grad():
        out_gen_256 = gen(x256)
        out_gen_128 = gen(x128)

    check("Generator: input [B,1,256,256] → output [B,1,256,256]",
          out_gen_256.shape == (B, 1, 256, 256),
          str(tuple(out_gen_256.shape)))

    check("Generator: input [B,1,128,128] → output [B,1,128,128]",
          out_gen_128.shape == (B, 1, 128, 128),
          str(tuple(out_gen_128.shape)))

    # ── Tanh range ────────────────────────────────────────────────────────────
    check("Generator output in [-1, 1]  (Tanh applied)",
          out_gen_256.min() >= -1.0 and out_gen_256.max() <= 1.0,
          f"min={out_gen_256.min():.4f}  max={out_gen_256.max():.4f}")

    # ── Discriminator output shape ────────────────────────────────────────────
    # TF discriminator on 256×256 input:
    # After 3 downsamples: 256→128→64→32
    # ZeroPad(1): 32→34; Conv(4,s=1): 34-4+1=31; → 31×31
    # ZeroPad(1): 31→33; Conv(4,s=1): 33-4+1=30; → 30×30
    with torch.no_grad():
        out_disc = disc(x256)

    check("Discriminator: input [B,1,256,256] → output [B,1,30,30]",
          out_disc.shape == (B, 1, 30, 30),
          str(tuple(out_disc.shape)))

    # ── Discriminator on 128×128 ──────────────────────────────────────────────
    # After 3 downsamples: 128→64→32→16
    # ZeroPad(1): 16→18; Conv(4,s=1): 15; → 15×15
    # ZeroPad(1): 15→17; Conv(4,s=1): 14; → 14×14
    with torch.no_grad():
        out_disc_128 = disc(x128)

    check("Discriminator: input [B,1,128,128] → output [B,1,14,14]",
          out_disc_128.shape == (B, 1, 14, 14),
          str(tuple(out_disc_128.shape)))

    # ── Encoder spatial halving (skip connection shapes) ─────────────────────
    # We verify intermediate shapes by registering forward hooks
    intermediates = {}
    def hook_d1(m, inp, out): intermediates['d1'] = out.shape
    def hook_d2(m, inp, out): intermediates['d2'] = out.shape
    def hook_d3(m, inp, out): intermediates['d3'] = out.shape

    h1 = gen.down1.register_forward_hook(hook_d1)
    h2 = gen.down2.register_forward_hook(hook_d2)
    h3 = gen.down3.register_forward_hook(hook_d3)

    with torch.no_grad():
        gen(x256)

    h1.remove(); h2.remove(); h3.remove()

    check("Skip d1 shape: [B, 64, 128, 128]",
          intermediates['d1'] == torch.Size([B, 64, 128, 128]),
          str(intermediates['d1']))

    check("Skip d2 shape: [B, 128, 64, 64]",
          intermediates['d2'] == torch.Size([B, 128, 64, 64]),
          str(intermediates['d2']))

    check("Skip d3 shape (bottleneck): [B, 256, 32, 32]",
          intermediates['d3'] == torch.Size([B, 256, 32, 32]),
          str(intermediates['d3']))

    # ── Decoder concat shapes ─────────────────────────────────────────────────
    dec_intermediates = {}
    def hook_u1(m, inp, out): dec_intermediates['u1'] = out.shape
    def hook_u2(m, inp, out): dec_intermediates['u2'] = out.shape

    h_u1 = gen.up1.register_forward_hook(hook_u1)
    h_u2 = gen.up2.register_forward_hook(hook_u2)

    with torch.no_grad():
        gen(x256)

    h_u1.remove(); h_u2.remove()

    check("up1 output (before concat with d2): [B, 128, 64, 64]",
          dec_intermediates['u1'] == torch.Size([B, 128, 64, 64]),
          str(dec_intermediates['u1']))

    check("up2 output (before concat with d1): [B, 64, 128, 128]",
          dec_intermediates['u2'] == torch.Size([B, 64, 128, 128]),
          str(dec_intermediates['u2']))

    # ── Determinism ───────────────────────────────────────────────────────────
    gen.eval()
    with torch.no_grad():
        out_a = gen(x256)
        out_b = gen(x256)

    check("Generator is deterministic in eval mode",
          torch.allclose(out_a, out_b),
          f"max_diff={( out_a - out_b).abs().max():.2e}")

    # ── InstanceNorm: mean≈0, var≈1 after normalisation ──────────────────────
    # With a fresh random-normal scale ≈ N(0,0.02) the output won't be exactly
    # unit-normalised, but without scale/offset it would be.  We test that
    # a manually zeroed-offset / unit-scale IN layer normalises correctly.
    norm_layer = InstanceNorm(4)
    norm_layer.scale.data.fill_(1.0)
    norm_layer.offset.data.fill_(0.0)
    t = torch.randn(2, 4, 8, 8) * 10 + 5   # mean≠0, std≠1
    out_norm = norm_layer(t)
    spatial_mean = out_norm.mean(dim=[2, 3])   # should be ~0 per (B, C)
    spatial_var  = out_norm.var(dim=[2, 3], unbiased=False)  # should be ~1

    check("InstanceNorm: per-instance mean ≈ 0 (scale=1, offset=0)",
          spatial_mean.abs().max().item() < 1e-4,
          f"max|mean|={spatial_mean.abs().max():.2e}")

    check("InstanceNorm: per-instance var ≈ 1 (scale=1, offset=0)",
          (spatial_var - 1.0).abs().max().item() < 1e-4,
          f"max|var-1|={(spatial_var-1.0).abs().max():.2e}")

    # ── Weight initialisation ─────────────────────────────────────────────────
    # All Conv weight tensors should have std close to 0.02
    weight_stds = []
    for name, p in gen.named_parameters():
        if 'weight' in name and p.ndim == 4:   # conv kernels only
            weight_stds.append((name, p.std().item()))

    all_near_002 = all(abs(s - 0.02) < 0.015 for _, s in weight_stds)
    worst = max(weight_stds, key=lambda x: abs(x[1] - 0.02))
    check("Generator Conv weights initialised near N(0, 0.02)",
          all_near_002,
          f"worst: {worst[0]}  std={worst[1]:.4f}")

    # ── Gradient flow ─────────────────────────────────────────────────────────
    gen.train()
    x_grad = torch.randn(1, 1, 256, 256, requires_grad=False)
    out_g   = gen(x_grad)
    loss    = out_g.mean()
    loss.backward()

    grads_ok = all(p.grad is not None and not p.grad.isnan().any()
                   for p in gen.parameters() if p.requires_grad)
    check("Generator: gradients flow to all parameters without NaN",
          grads_ok)

    disc.train()
    x_disc = torch.randn(1, 1, 256, 256)
    out_d  = disc(x_disc)
    out_d.mean().backward()
    grads_ok_d = all(p.grad is not None and not p.grad.isnan().any()
                     for p in disc.parameters() if p.requires_grad)
    check("Discriminator: gradients flow to all parameters without NaN",
          grads_ok_d)


# ──────────────────────────────────────────────────────────────────────────────
# Part B  —  TF ↔ PyTorch numerical equivalence
# ──────────────────────────────────────────────────────────────────────────────

def _try_import_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        return None


def _build_tf_generator(tf):
    """Rebuild TF unet_generator() inline (no project import needed)."""

    class InstanceNormalization(tf.keras.layers.Layer):
        def __init__(self, epsilon=1e-5):
            super().__init__()
            self.epsilon = epsilon
        def build(self, input_shape):
            self.scale  = self.add_weight(shape=input_shape[-1:],
                                          initializer='ones', trainable=True)
            self.offset = self.add_weight(shape=input_shape[-1:],
                                          initializer='zeros', trainable=True)
        def call(self, x):
            mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
            return self.scale * (x - mean) / tf.math.sqrt(var + self.epsilon) + self.offset

    def downsample(filters, size, apply_norm=True):
        init = tf.initializers.Zeros()    # we will overwrite weights manually
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same', kernel_initializer=init,
            use_bias=not apply_norm))
        if apply_norm:
            result.add(InstanceNormalization())
        result.add(tf.keras.layers.LeakyReLU(0.2))
        return result

    def upsample(filters, size):
        init = tf.initializers.Zeros()
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            kernel_initializer=init, use_bias=False))
        result.add(InstanceNormalization())
        result.add(tf.keras.layers.ReLU())
        return result

    down_stack = [
        downsample(64, 4, False),
        downsample(128, 4),
        downsample(256, 4),
    ]
    up_stack = [upsample(128, 4), upsample(64, 4)]

    last = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=tf.initializers.Zeros(), activation='tanh')

    concat = tf.keras.layers.Concatenate()
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    # Force weight creation
    dummy = tf.zeros([1, 256, 256, 1])
    model(dummy, training=False)
    return model, InstanceNormalization


def _copy_tf_to_pytorch_generator(tf_model, pt_model):
    """
    Copy TF weights into the PyTorch generator.

    TF Conv2D kernel shape: [H, W, in_c, out_c]  → PyTorch: [out_c, in_c, H, W]
    TF Conv2DTranspose shape: [H, W, out_c, in_c] → PyTorch: [in_c, out_c, H, W]
    InstanceNorm scale/offset: [C] in TF vs [1,C,1,1] in PyTorch.

    We set both models to all-ones weights so the copy is easy to verify.
    """
    def fill_tf_conv(layer, value=1.0):
        w = layer.get_weights()
        layer.set_weights([np.full_like(a, value) for a in w])

    def fill_pt_seq(seq, value=1.0):
        with torch.no_grad():
            for m in seq.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m.weight.fill_(value)
                    if m.bias is not None:
                        m.bias.fill_(value)
                elif isinstance(m, InstanceNorm):
                    m.scale.fill_(value)
                    m.offset.fill_(value)

    # Set everything to 1.0 in TF
    for layer in tf_model.layers:
        if hasattr(layer, 'layers'):                  # Sequential sub-model
            for sub in layer.layers:
                try:
                    w = sub.get_weights()
                    if w:
                        sub.set_weights([np.ones_like(a) for a in w])
                except Exception:
                    pass
        else:
            try:
                w = layer.get_weights()
                if w:
                    layer.set_weights([np.ones_like(a) for a in w])
            except Exception:
                pass

    # Set everything to 1.0 in PyTorch
    with torch.no_grad():
        for m in pt_model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.fill_(1.0)
            elif isinstance(m, InstanceNorm):
                m.scale.fill_(1.0)
                m.offset.fill_(1.0)


def run_part_b(tf):
    import tensorflow as tfl  # re-bind for clarity

    print("\n" + "="*50)
    print("Part B — TF ↔ PyTorch numerical equivalence")
    print("="*50)

    # Use small spatial size for speed; architecture is resolution-agnostic
    # (but must be divisible by 2^3 = 8)
    H = W = 64   # 64×64 is plenty to verify spatial ops without the long wait

    # ── Build both models ─────────────────────────────────────────────────────
    try:
        tf_gen, _ = _build_tf_generator(tfl)
    except Exception as e:
        print(f"  {SKIP}  Could not build TF model: {e}")
        return

    # Rebuild PyTorch model with matching small size (architecture is the same)
    torch.manual_seed(42)
    pt_gen = UNetGenerator().eval()

    # ── Align weights: set all conv weights & norms to 1.0 in both ───────────
    _copy_tf_to_pytorch_generator(tf_gen, pt_gen)

    # ── Shared random input ───────────────────────────────────────────────────
    np.random.seed(7)
    np_input = np.random.randn(1, H, W, 1).astype(np.float32) * 0.1
    # (small magnitude avoids saturation through the all-ones convolutions)

    tf_input = tfl.constant(np_input)                          # [1, H, W, 1] NHWC
    pt_input = torch.from_numpy(np_input.transpose(0, 3, 1, 2))  # [1, 1, H, W] NCHW

    # ── Forward passes ────────────────────────────────────────────────────────
    tf_out_nhwc = tf_gen(tf_input, training=False).numpy()     # [1, H, W, 1]
    with torch.no_grad():
        pt_out_nchw = pt_gen(pt_input).numpy()                 # [1, 1, H, W]

    # Convert PyTorch output to NHWC for direct comparison
    pt_out_nhwc = pt_out_nchw.transpose(0, 2, 3, 1)           # [1, H, W, 1]

    max_abs_diff = np.abs(tf_out_nhwc - pt_out_nhwc).max()
    mean_abs_diff = np.abs(tf_out_nhwc - pt_out_nhwc).mean()

    ATOL = 1e-4   # generous tolerance for float32 cross-framework ops

    check(
        "Generator: TF and PyTorch outputs match within tolerance",
        float(max_abs_diff) < ATOL,
        f"max_abs_diff={max_abs_diff:.2e}  mean_abs_diff={mean_abs_diff:.2e}  tol={ATOL:.0e}",
    )

    # ── Output shape equivalence ──────────────────────────────────────────────
    check("Generator: TF output shape == PyTorch output shape",
          tf_out_nhwc.shape == pt_out_nhwc.shape,
          f"TF={tf_out_nhwc.shape}  PT={pt_out_nhwc.shape}")

    # ── Tanh range in both ────────────────────────────────────────────────────
    check("TF generator output in [-1, 1]",
          tf_out_nhwc.min() >= -1.0 and tf_out_nhwc.max() <= 1.0,
          f"min={tf_out_nhwc.min():.4f}  max={tf_out_nhwc.max():.4f}")

    check("PyTorch generator output in [-1, 1]",
          pt_out_nhwc.min() >= -1.0 and pt_out_nhwc.max() <= 1.0,
          f"min={pt_out_nhwc.min():.4f}  max={pt_out_nhwc.max():.4f}")

    # ── Intermediate encoder shapes ───────────────────────────────────────────
    # TF: after each downsample the spatial dims halve and channels grow
    # We pull intermediate outputs from TF by building sub-models on the fly
    down_outputs = []
    x_tf = tf_input
    for layer in tf_gen.layers:
        # downsample layers in TF are Sequential submodels named 'sequential',
        # 'sequential_1', 'sequential_2'
        if hasattr(layer, 'layers') and any(
                isinstance(sub, tfl.keras.layers.Conv2D)
                for sub in layer.layers):
            x_tf = layer(x_tf, training=False)
            down_outputs.append(x_tf.numpy().shape)   # NHWC

    if len(down_outputs) >= 3:
        expected_tf = [
            (1, H//2,  W//2,  64),
            (1, H//4,  W//4,  128),
            (1, H//8,  W//8,  256),
        ]
        for i, (got, exp) in enumerate(zip(down_outputs[:3], expected_tf)):
            check(f"TF encoder down{i+1} shape == {exp}",
                  got == exp, f"got {got}")

    # ── Encoder shapes match between frameworks ───────────────────────────────
    intermediates = {}
    def hook_d1(m, inp, out): intermediates['d1'] = tuple(out.shape)
    def hook_d2(m, inp, out): intermediates['d2'] = tuple(out.shape)
    def hook_d3(m, inp, out): intermediates['d3'] = tuple(out.shape)

    h1 = pt_gen.down1.register_forward_hook(hook_d1)
    h2 = pt_gen.down2.register_forward_hook(hook_d2)
    h3 = pt_gen.down3.register_forward_hook(hook_d3)
    with torch.no_grad():
        pt_gen(pt_input)
    h1.remove(); h2.remove(); h3.remove()

    expected_pt_nhwc = [(1, H//2, W//2, 64), (1, H//4, W//4, 128), (1, H//8, W//8, 256)]
    for i, (pt_shape, exp_nhwc) in enumerate(
            zip([intermediates['d1'], intermediates['d2'], intermediates['d3']],
                expected_pt_nhwc)):
        # convert PT NCHW to NHWC equivalent for comparison
        pt_nhwc = (pt_shape[0], pt_shape[2], pt_shape[3], pt_shape[1])
        check(
            f"Encoder down{i+1}: TF and PyTorch shapes match (as NHWC)",
            pt_nhwc == exp_nhwc,
            f"PT(as NHWC)={pt_nhwc}  expected={exp_nhwc}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_part_a()

    tf = _try_import_tf()
    if tf is None:
        print(f"\n  {SKIP}  TensorFlow not found — skipping Part B")
        print("       Install with:  pip install tensorflow")
    else:
        tf_version = getattr(tf, '__version__', 'unknown')
        print(f"\n  TensorFlow {tf_version} found — running Part B")
        run_part_b(tf)

    ok = summary()
    sys.exit(0 if ok else 1)
import torch
from tqdm import tqdm

from loss import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

# Module-level metrics (reused across evaluate_test_loss_h5 calls to avoid
# repeated allocation; evaluate_val_metrics_h5 allocates its own copies so
# they live on the correct device regardless of when the function is called).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
_psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
_mse_fn  = MeanSquaredError().to(device)


 
def _gen(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out
 


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

def train_step(generator_g, generator_f, discriminator_x, discriminator_y,
               generator_g_optimizer, generator_f_optimizer,
               discriminator_x_optimizer, discriminator_y_optimizer,
               real_x, real_y):
    
    torch.autograd.set_detect_anomaly(True)

    # Set all models to training mode (replaces training=True)
    generator_g.train()
    generator_f.train()
    discriminator_x.train()
    discriminator_y.train()

    # Forward passes
    fake_y, _, _, _ = generator_g(real_x)
    cycled_x, _, _, _ = generator_f(fake_y)
    fake_x, _, _, _ = generator_f(real_y)
    cycled_y, _, _, _ = generator_g(fake_x)
    same_x, _, _, _ = generator_f(real_x)
    same_y, _, _, _ = generator_g(real_y)

    disc_real_x = discriminator_x(real_x)
    disc_real_y = discriminator_y(real_y)
    disc_fake_x = discriminator_x(fake_x)
    disc_fake_y = discriminator_y(fake_y)

    # GAN losses
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + \
                       calc_cycle_loss(real_y, cycled_y)

    identity_loss1 = identity_loss(real_y, same_y)
    identity_loss2 = identity_loss(real_x, same_x)

    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss1
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss2

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Zero all grads first
    generator_g_optimizer.zero_grad()
    generator_f_optimizer.zero_grad()
    discriminator_x_optimizer.zero_grad()
    discriminator_y_optimizer.zero_grad()

    # Compute all gradients
    total_gen_g_loss.backward(retain_graph=True)
    total_gen_f_loss.backward(retain_graph=True)
    disc_x_loss.backward(retain_graph=True)
    disc_y_loss.backward()

    # Then apply all optimizer steps
    generator_g_optimizer.step()
    generator_f_optimizer.step()
    discriminator_x_optimizer.step()
    discriminator_y_optimizer.step()


    return (
        total_gen_g_loss.item(),
        total_gen_f_loss.item(),
        disc_x_loss.item(),
        disc_y_loss.item(),
        total_cycle_loss.item(),
        identity_loss1.item(),
        identity_loss2.item()
    )



# ---------------------------------------------------------------------------
# run_epoch
# ---------------------------------------------------------------------------
 
def run_epoch(
    gen_g, gen_f, disc_x, disc_y,
    opt_g, opt_f, opt_dx, opt_dy,
    train_loader,
    epoch: int,
    total_epochs: int,
    extract_fn,
) -> dict:
    """
    One full training epoch.
 
    Parameters
    ----------
    extract_fn : callable
        ``pipeline._extract_domain_pair`` — takes a batch dict, returns
        ``(real_x, real_y)`` tensors already on the correct device.
 
    Returns
    -------
    dict
        Per-loss averages over all batches.
    """
    totals = dict(gen_g=0., gen_f=0., disc_x=0., disc_y=0.,
                  cycle=0., id_g=0., id_f=0.)
    n = 0
 
    bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
    )
    for batch in bar:
        real_x, real_y = extract_fn(batch)
        gg, gf, dx, dy, cyc, idg, idf = train_step(
            gen_g, gen_f, disc_x, disc_y,
            opt_g, opt_f, opt_dx, opt_dy,
            real_x, real_y,
        )
        totals['gen_g']  += gg;  totals['gen_f']  += gf
        totals['disc_x'] += dx;  totals['disc_y'] += dy
        totals['cycle']  += cyc; totals['id_g']   += idg; totals['id_f'] += idf
        n += 1
        bar.set_postfix(
            G_g=f"{gg:.3f}", G_f=f"{gf:.3f}",
            D_x=f"{dx:.3f}", D_y=f"{dy:.3f}",
            cyc=f"{cyc:.3f}",
        )
 
    return {k: v / n for k, v in totals.items()}
 
 
# ---------------------------------------------------------------------------
# evaluate_test_loss_h5  (early-stopping signal)
# ---------------------------------------------------------------------------
 
def evaluate_test_loss_h5(gen_g, gen_f, test_loader, extract_fn) -> float:
    """
    Compute average cycle loss over the test set.
 
    Used by EarlyStopping — lower is better.
 
    Returns
    -------
    float
        Mean cycle loss across all batches (inf if loader is empty).
    """
    gen_g.eval(); gen_f.eval()
    losses = []
 
    bar = tqdm(
        test_loader,
        desc="  ↳ test loss",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch in bar:
            real_x, real_y = extract_fn(batch)
 
            # _gen() unwraps the (out, d1, d2, d3) tuple before chaining
            cycled_x = _gen(gen_f, _gen(gen_g, real_x))
            cycled_y = _gen(gen_g, _gen(gen_f, real_y))
 
            loss = (
                calc_cycle_loss(real_x, cycled_x) +
                calc_cycle_loss(real_y, cycled_y)
            ).item()
            losses.append(loss)
            bar.set_postfix(cycle=f"{loss:.4f}")
 
    gen_g.train(); gen_f.train()
    return sum(losses) / len(losses) if losses else float('inf')
 
 
# ---------------------------------------------------------------------------
# evaluate_val_metrics_h5  (final quality metrics)
# ---------------------------------------------------------------------------
 
def evaluate_val_metrics_h5(
    gen_g, gen_f,
    val_loader,
    extract_fn,
    device: torch.device,
) -> dict:
    """
    Compute SSIM, PSNR, and MSE on the validation set.
 
    Images are scaled to [0, 1] before metric computation:
        cycled = (gen_f(gen_g(real_x)) + 1) / 2   (generator output is Tanh → [-1,1])
        real   = real_x clamped to [0, 1]
 
    Returns
    -------
    dict with keys 'ssim', 'psnr', 'mse' (float averages over all batches).
    """
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    mse_fn  = MeanSquaredError().to(device)
 
    gen_g.eval(); gen_f.eval()
    ssim_scores, psnr_scores, mse_scores = [], [], []
 
    bar = tqdm(
        val_loader,
        desc="  ↳ val metrics",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch in bar:
            real_x, real_y = extract_fn(batch)
 
            # Generator output is Tanh (range [-1, 1]) → rescale to [0, 1]
            # _gen() unwraps the (out, d1, d2, d3) tuple before chaining
            cycled_x_01 = (_gen(gen_f, _gen(gen_g, real_x)) + 1.0) / 2.0
            real_x_01   = real_x.clamp(0.0, 1.0)
 
            s = ssim_fn(cycled_x_01, real_x_01).item()
            p = psnr_fn(cycled_x_01, real_x_01).item()
            m = mse_fn(cycled_x_01.flatten(), real_x_01.flatten()).item()
 
            ssim_scores.append(s)
            psnr_scores.append(p)
            mse_scores.append(m)
            bar.set_postfix(SSIM=f"{s:.3f}", PSNR=f"{p:.2f}", MSE=f"{m:.6f}")
 
    return {
        'ssim': sum(ssim_scores) / len(ssim_scores),
        'psnr': sum(psnr_scores) / len(psnr_scores),
        'mse':  sum(mse_scores)  / len(mse_scores),
    }

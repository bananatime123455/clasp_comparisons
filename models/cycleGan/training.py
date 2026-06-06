import torch
from tqdm import tqdm
from loss import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
mse_fn  = MeanSquaredError().to(device)


def train_step(gen_g, gen_f, disc_x, disc_y,
               opt_g, opt_f, opt_dx, opt_dy,
               real_x, real_y):

    gen_g.train(); gen_f.train()
    disc_x.train(); disc_y.train()

    fake_y   = gen_g(real_x)
    cycled_x = gen_f(fake_y)
    fake_x   = gen_f(real_y)
    cycled_y = gen_g(fake_x)
    same_x   = gen_f(real_x)
    same_y   = gen_g(real_y)

    cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    id_loss1   = identity_loss(real_y, same_y)
    id_loss2   = identity_loss(real_x, same_x)
    gen_g_loss = generator_loss(disc_y(fake_y)) + cycle_loss + id_loss1
    gen_f_loss = generator_loss(disc_x(fake_x)) + cycle_loss + id_loss2

    opt_g.zero_grad(); opt_f.zero_grad()
    gen_g_loss.backward(retain_graph=True)
    gen_f_loss.backward()
    opt_g.step(); opt_f.step()

    disc_x_loss = discriminator_loss(disc_x(real_x), disc_x(fake_x.detach()))
    disc_y_loss = discriminator_loss(disc_y(real_y), disc_y(fake_y.detach()))

    opt_dx.zero_grad(); opt_dy.zero_grad()
    disc_x_loss.backward()
    disc_y_loss.backward()
    opt_dx.step(); opt_dy.step()

    return (gen_g_loss.item(), gen_f_loss.item(),
            disc_x_loss.item(), disc_y_loss.item(),
            cycle_loss.item(), id_loss1.item(), id_loss2.item())


def run_epoch(gen_g, gen_f, disc_x, disc_y,
              opt_g, opt_f, opt_dx, opt_dy,
              train_loader, epoch, total_epochs, extract_fn):
    """One full training epoch with a per-batch tqdm bar."""
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
        bar.set_postfix(G_g=f"{gg:.3f}", G_f=f"{gf:.3f}",
                        D_x=f"{dx:.3f}", D_y=f"{dy:.3f}", cyc=f"{cyc:.3f}")

    return {k: v / n for k, v in totals.items()}


def evaluate_test_loss_h5(gen_g, gen_f, test_loader, extract_fn):
    """Cycle loss on test set with progress bar — early-stopping signal."""
    gen_g.eval(); gen_f.eval()
    losses = []

    bar = tqdm(test_loader, desc="  ↳ test loss", unit="batch",
               leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in bar:
            real_x, real_y = extract_fn(batch)
            cycled_x = gen_f(gen_g(real_x))
            cycled_y = gen_g(gen_f(real_y))
            loss = (calc_cycle_loss(real_x, cycled_x) +
                    calc_cycle_loss(real_y, cycled_y)).item()
            losses.append(loss)
            bar.set_postfix(cycle=f"{loss:.4f}")

    gen_g.train(); gen_f.train()
    return sum(losses) / len(losses) if losses else float('inf')


def evaluate_val_metrics_h5(gen_g, gen_f, val_loader, extract_fn, device):
    """SSIM / PSNR / MSE on val set with progress bar — final metrics only."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from torchmetrics.regression import MeanSquaredError

    _ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    _psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    _mse  = MeanSquaredError().to(device)

    gen_g.eval(); gen_f.eval()
    ssim_scores, psnr_scores, mse_scores = [], [], []

    bar = tqdm(val_loader, desc="  ↳ val metrics", unit="batch",
               leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in bar:
            real_x, real_y = extract_fn(batch)
            cycled_x_01 = (gen_f(gen_g(real_x)) + 1.0) / 2.0
            real_x_01   = real_x.clamp(0.0, 1.0)

            s = _ssim(cycled_x_01, real_x_01).item()
            p = _psnr(cycled_x_01, real_x_01).item()
            m = _mse( cycled_x_01, real_x_01).item()
            ssim_scores.append(s); psnr_scores.append(p); mse_scores.append(m)
            bar.set_postfix(SSIM=f"{s:.3f}", PSNR=f"{p:.2f}", MSE=f"{m:.6f}")

    return {
        'ssim': sum(ssim_scores) / len(ssim_scores),
        'psnr': sum(psnr_scores) / len(psnr_scores),
        'mse':  sum(mse_scores)  / len(mse_scores),
    }
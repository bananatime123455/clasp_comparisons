import torch
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


def evaluate_test_loss(gen_g, gen_f, loader_x, loader_y, device):
    """Cycle loss on test set — used only for early stopping signal during training."""
    gen_g.eval(); gen_f.eval()
    losses = []
    with torch.no_grad():
        for real_x, real_y in zip(loader_x, loader_y):
            real_x = real_x.to(device)
            real_y = real_y.to(device)
            cycled_x = gen_f(gen_g(real_x))
            cycled_y = gen_g(gen_f(real_y))
            losses.append(
                (calc_cycle_loss(real_x, cycled_x) +
                 calc_cycle_loss(real_y, cycled_y)).item()
            )
    return sum(losses) / len(losses)


def evaluate_val_metrics(gen_g, gen_f, loader_x, loader_y, device):
    """SSIM, PSNR, MSE on val set — final reported metrics only."""
    gen_g.eval(); gen_f.eval()

    ssim_scores, psnr_scores, mse_scores = [], [], []

    with torch.no_grad():
        for real_x, real_y in zip(loader_x, loader_y):
            real_x = real_x.to(device)
            real_y = real_y.to(device)

            fake_y   = gen_g(real_x)
            cycled_x = gen_f(fake_y)

            # Rescale [-1, 1] → [0, 1] for metrics
            real_x_01   = (real_x   + 1) / 2
            cycled_x_01 = (cycled_x + 1) / 2

            ssim_scores.append(ssim_fn(cycled_x_01, real_x_01).item())
            psnr_scores.append(psnr_fn(cycled_x_01, real_x_01).item())
            mse_scores.append( mse_fn( cycled_x_01, real_x_01).item())

    return {
        'ssim': sum(ssim_scores) / len(ssim_scores),
        'psnr': sum(psnr_scores) / len(psnr_scores),
        'mse':  sum(mse_scores)  / len(mse_scores),
    }
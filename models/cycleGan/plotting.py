import pathlib
import numpy as np
import matplotlib.pyplot as plt


def generate_images(generator_g, test1, generator_f, test2, epoch, output_dir):
    """Save a 2x2 grid of {input, translated} images for both directions."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction1 = generator_g(test1, training=False)
    prediction2 = generator_f(test2, training=False)
    plt.figure(figsize=(8, 8))
    display_list = [test1[0], prediction1[0], test2[0], prediction2[0]]
    title = ['Input Mod 1 Image', 'Translated to  Mod 2',
             'Input Mod 2 Image', 'Translated to Mod 1']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        # Images are in [-1, 1] (tanh range) -> rescale to [0, 1] for display.
        plt.imshow(display_list[i].numpy()[:, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    save_path = output_dir / f'image_at_epoch_{epoch:04d}.png'
    plt.savefig(save_path)
    plt.close()


def plot_losses(gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss, output_dir,
                 cycle_loss=None, identity_loss1=None, identity_loss2=None):
    """Plot generator/discriminator loss curves (and optional component
    losses) over training epochs and save to output_dir/loss_curves.png."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(gen_g_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gen_g_loss, label='Generator G loss')
    plt.plot(epochs, gen_f_loss, label='Generator F loss')
    plt.plot(epochs, disc_x_loss, label='Discriminator X loss')
    plt.plot(epochs, disc_y_loss, label='Discriminator Y loss')

    if cycle_loss is not None and any(cycle_loss):
        plt.plot(epochs, cycle_loss, label='Cycle loss', linestyle='--')
    if identity_loss1 is not None and any(identity_loss1):
        plt.plot(epochs, identity_loss1, label='Identity loss (X)', linestyle=':')
    if identity_loss2 is not None and any(identity_loss2):
        plt.plot(epochs, identity_loss2, label='Identity loss (Y)', linestyle=':')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CycleGAN Training Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png')
    plt.close()


def plot_metrics(ssim_vals, psnr_vals, mse_vals, output_dir):
    """
    Plot distributions of the per-slice SSIM, PSNR, and MSE computed between
    reconstructed modality-2 images and the real modality-2 images, and save
    to output_dir/eval_metrics.png. Also prints summary means.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(ssim_vals, bins=20, color='tab:blue')
    axes[0].set_title(f'SSIM (mean={np.mean(ssim_vals):.4f})')
    axes[0].set_xlabel('SSIM')

    axes[1].hist(psnr_vals, bins=20, color='tab:orange')
    axes[1].set_title(f'PSNR (mean={np.mean(psnr_vals):.4f} dB)')
    axes[1].set_xlabel('PSNR (dB)')

    axes[2].hist(mse_vals, bins=20, color='tab:green')
    axes[2].set_title(f'MSE (mean={np.mean(mse_vals):.6f})')
    axes[2].set_xlabel('MSE')

    plt.tight_layout()
    plt.savefig(output_dir / 'eval_metrics.png')
    plt.close()

    print(
        f"Eval metrics — SSIM: {np.mean(ssim_vals):.4f} | "
        f"PSNR: {np.mean(psnr_vals):.4f} dB | MSE: {np.mean(mse_vals):.6f}"
    )
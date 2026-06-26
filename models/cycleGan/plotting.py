import matplotlib.pyplot as plt
import numpy as np


def generate_images(generator_g, test1, generator_f, test2, epoch, data_dir_results):
    prediction1 = generator_g(test1, training=False)
    prediction2 = generator_f(test2, training=False)

    def to_plot(tensor):
        """Return the first image in the batch as a (H, W) numpy array."""
        return tensor[0, :, :, 0].numpy()

    plt.figure(figsize=(8, 8))
    display_list = [
        to_plot(test1),
        to_plot(prediction1),
        to_plot(test2),
        to_plot(prediction2),
    ]
    title = ['Input T1 Image', 'Translated to T2', 'Input T2 Image', 'Translated to T1']

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    plt.savefig(data_dir_results / 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def plot_losses(gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss,
                data_dir_results, cycle_loss, identity_loss1, identity_loss2):

    epochs = np.arange(len(gen_g_loss))

    fig, axs = plt.subplots(2, 4, figsize=(14, 8))

    axs[0, 0].plot(epochs, gen_g_loss)
    axs[0, 0].set_title("Generator G Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, gen_f_loss)
    axs[0, 1].set_title("Generator F Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True)

    axs[0, 2].plot(epochs, disc_x_loss)
    axs[0, 2].set_title("Discriminator X Loss")
    axs[0, 2].set_xlabel("Epoch")
    axs[0, 2].set_ylabel("Loss")
    axs[0, 2].grid(True)

    axs[0, 3].plot(epochs, disc_y_loss)
    axs[0, 3].set_title("Discriminator Y Loss")
    axs[0, 3].set_xlabel("Epoch")
    axs[0, 3].set_ylabel("Loss")
    axs[0, 3].grid(True)

    axs[1, 0].plot(epochs, cycle_loss)
    axs[1, 0].set_title("Cycle Consistency Loss")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, identity_loss1)
    axs[1, 1].set_title("Identity Loss A to B")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].grid(True)

    axs[1, 2].plot(epochs, identity_loss2)
    axs[1, 2].set_title("Identity Loss B to A")
    axs[1, 2].set_xlabel("Epoch")
    axs[1, 2].set_ylabel("Loss")
    axs[1, 2].grid(True)

    # Leave axs[1,3] blank so layout stays consistent
    axs[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(data_dir_results / "losses.png")
    plt.close()


def plot_metrics(ssim_vals, psnr_vals, mse_vals, data_dir_results):

    epochs = np.arange(len(ssim_vals))

    avg_ssim = np.mean(ssim_vals)
    avg_psnr = np.mean(psnr_vals)
    avg_mse  = np.mean(mse_vals)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(epochs, ssim_vals, label="SSIM")
    axs[0].axhline(avg_ssim, linestyle="--", label=f"Avg: {avg_ssim:.4f}")
    axs[0].set_title("SSIM vs Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("SSIM")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, psnr_vals, label="PSNR")
    axs[1].axhline(avg_psnr, linestyle="--", label=f"Avg: {avg_psnr:.2f}")
    axs[1].set_title("PSNR vs Epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PSNR")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(epochs, mse_vals, label="MSE")
    axs[2].axhline(avg_mse, linestyle="--", label=f"Avg: {avg_mse:.6f}")
    axs[2].set_title("MSE vs Epoch")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("MSE")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(data_dir_results / "metrics.png")
    plt.close()
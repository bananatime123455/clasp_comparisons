import pathlib
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from data_processing import load_datasets
from mri_gan import unet_generator, discriminator
from loss import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss
from training import train_step
from plotting import generate_images, plot_losses, plot_metrics
from early_stopping import EarlyStopping

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False

IMG_HEIGHT = 256
IMG_WIDTH  = 256


class MRIGANPipeline:
    def __init__(
        self,
        train_files,
        test_files,
        val_files,
        modality_1,
        modality_2,
        batch_size,
        buffer_size,
        epochs,
        early_stopping_patience,
        early_stopping_delta,
        data_dir_results
    ):
        self.modality_1       = modality_1
        self.modality_2       = modality_2
        self.batch_size       = batch_size
        self.buffer_size      = buffer_size
        self.epochs           = epochs
        self.data_dir_results = pathlib.Path(data_dir_results)
        self.data_dir_results.mkdir(parents=True, exist_ok=True)

        # ── Models ──────────────────────────────────────────────────────────
        self.generator_g     = unet_generator()
        self.generator_f     = unet_generator()
        self.discriminator_x = discriminator()
        self.discriminator_y = discriminator()

        # ── Optimizers ──────────────────────────────────────────────────────
        self.generator_g_optimizer      = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer      = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # ── Datasets ────────────────────────────────────────────────────────
        (
            self.tr1_train, self.tr2_train,
            self.tr1_test,  self.tr2_test,
            self.tr1_val,   self.tr2_val,
        ) = load_datasets(
            train_files, test_files, val_files,
            modality_1, modality_2,
            IMG_HEIGHT, IMG_WIDTH,
            batch_size, buffer_size,
        )

        # ── Early stopping ───────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=early_stopping_delta,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _models_dict(self):
        return {
            "generator_g":     self.generator_g,
            "generator_f":     self.generator_f,
            "discriminator_x": self.discriminator_x,
            "discriminator_y": self.discriminator_y,
        }

    def _train_epoch(self):
        """Run one full pass over the training set and return mean losses."""
        g_g, g_f, d_x, d_y = [], [], [], []

        dataset = tf.data.Dataset.zip((self.tr1_train, self.tr2_train))
        total = sum(1 for _ in dataset)  # get step count for the progress bar

        pbar = tqdm(zip(self.tr1_train, self.tr2_train), total=total, desc="  Training", unit="step")

        for image_x, image_y in pbar:
            gg, gf, dx, dy = train_step(
                self.generator_g, self.generator_f,
                self.discriminator_x, self.discriminator_y,
                self.generator_g_optimizer, self.generator_f_optimizer,
                self.discriminator_x_optimizer, self.discriminator_y_optimizer,
                image_x, image_y,
            )
            g_g.append(float(gg))
            g_f.append(float(gf))
            d_x.append(float(dx))
            d_y.append(float(dy))

            pbar.set_postfix({
                "G_g": f"{float(gg):.4f}",
                "G_f": f"{float(gf):.4f}",
                "D_x": f"{float(dx):.4f}",
                "D_y": f"{float(dy):.4f}",
            })

        return np.mean(g_g), np.mean(g_f), np.mean(d_x), np.mean(d_y)

    def _val_loss(self):
        """Compute mean generator loss over the validation set."""
        losses = []
        for image_x, image_y in tf.data.Dataset.zip((self.tr1_val, self.tr2_val)):
            fake_y      = self.generator_g(image_x, training=False)
            fake_x      = self.generator_f(image_y, training=False)
            disc_fake_y = self.discriminator_y(fake_y, training=False)
            disc_fake_x = self.discriminator_x(fake_x, training=False)
            loss = float(generator_loss(disc_fake_y)) + float(generator_loss(disc_fake_x))
            losses.append(loss)
        return float(np.mean(losses)) if losses else 0.0

    def _compute_metrics(self, dataset_x, dataset_y):
        """
        Compute SSIM, PSNR, and MSE of generator_g predictions against
        real modality_2 images on the provided dataset pair.
        """
        ssim_vals, psnr_vals, mse_vals = [], [], []

        for real_x, real_y in tf.data.Dataset.zip((dataset_x, dataset_y)):
            fake_y = self.generator_g(real_x, training=False).numpy()
            real_y_np = real_y.numpy()

            for i in range(real_x.shape[0]):
                pred  = fake_y[i, :, :, 0]
                truth = real_y_np[i, :, :, 0]

                mse = float(np.mean((pred - truth) ** 2))
                mse_vals.append(mse)

                if _SKIMAGE_OK:
                    ssim_vals.append(float(_ssim_fn(truth, pred, data_range=1.0)))
                    psnr_vals.append(float(_psnr_fn(truth, pred, data_range=1.0)))

        return (
            ssim_vals if ssim_vals else [0.0],
            psnr_vals if psnr_vals else [0.0],
            mse_vals  if mse_vals  else [0.0],
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def train(self):
        """
        Train the CycleGAN for self.epochs epochs with early stopping.

        Returns a results dict compatible with the comp_pipeline caller:
        {
            "model":          "cycle_GAN",
            "modality_1":     str,
            "modality_2":     str,
            "epochs_trained": int,
            "gen_g_loss":     [float, ...],
            "gen_f_loss":     [float, ...],
            "disc_x_loss":    [float, ...],
            "disc_y_loss":    [float, ...],
            "val_loss":       [float, ...],
            "ssim":           float,   # mean over test set (last epoch)
            "psnr":           float,
            "mse":            float,
        }
        """
        history = {
            "gen_g_loss":  [],
            "gen_f_loss":  [],
            "disc_x_loss": [],
            "disc_y_loss": [],
            "val_loss":    [],
        }

        epochs_trained = 0

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            # ── Train ───────────────────────────────────────────────────────
            gg, gf, dx, dy = self._train_epoch()
            history["gen_g_loss"].append(gg)
            history["gen_f_loss"].append(gf)
            history["disc_x_loss"].append(dx)
            history["disc_y_loss"].append(dy)

            print(
                f"  Gen G: {gg:.4f} | Gen F: {gf:.4f} | "
                f"Disc X: {dx:.4f} | Disc Y: {dy:.4f}"
            )

            # ── Validation loss + early stopping ────────────────────────────
            val_loss = self._val_loss()
            history["val_loss"].append(val_loss)
            print(f"  Val loss: {val_loss:.4f}")

            self.early_stopping(val_loss, self._models_dict())
            epochs_trained = epoch + 1

            # ── Visualise sample outputs ─────────────────────────────────────
            generate_images(
                self.generator_g, next(iter(self.tr1_train)),
                self.generator_f, next(iter(self.tr2_train)),
                epoch, self.data_dir_results,
            )

            if self.early_stopping.early_stop:
                print(f"  Early stopping triggered after epoch {epochs_trained}.")
                break

        # Restore best weights
        self.early_stopping.load_best_weights(self._models_dict())

        # ── Final metrics on test set ────────────────────────────────────────
        ssim_vals, psnr_vals, mse_vals = self._compute_metrics(
            self.tr1_test, self.tr2_test
        )

        n = len(history["gen_g_loss"])
        plot_losses(
            history["gen_g_loss"],
            history["gen_f_loss"],
            history["disc_x_loss"],
            history["disc_y_loss"],
            self.data_dir_results,
            cycle_loss=[0.0] * n,
            identity_loss1=[0.0] * n,
            identity_loss2=[0.0] * n,
        )
        plot_metrics(ssim_vals, psnr_vals, mse_vals, self.data_dir_results)

        print("\nTraining completed!")

        return {
            "model":          "cycle_GAN",
            "modality_1":     self.modality_1,
            "modality_2":     self.modality_2,
            "epochs_trained": epochs_trained,
            **history,
            "ssim": float(np.mean(ssim_vals)),
            "psnr": float(np.mean(psnr_vals)),
            "mse":  float(np.mean(mse_vals)),
        }
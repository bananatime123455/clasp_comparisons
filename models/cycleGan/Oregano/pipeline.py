import pathlib
import numpy as np
import tensorflow as tf
import torch

from tqdm import tqdm
from data_processing import load_datasets
from mri_gan import unet_generator, discriminator
from loss import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss
from training import train_step
from plotting import generate_images, plot_losses, plot_metrics
from early_stopping import EarlyStopping

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.regression import MeanSquaredError

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

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def _models_dict(self):
        return {
            "generator_g":     self.generator_g,
            "generator_f":     self.generator_f,
            "discriminator_x": self.discriminator_x,
            "discriminator_y": self.discriminator_y,
        }


    def train_epoch(self):
        # Loop through the dataset and perform training step
        for image_x, image_y in tf.data.Dataset.zip((self.tr1_train, self.tr2_train)):
            gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = train_step(
                self.generator_g, self.generator_f,
                self.discriminator_x, self.discriminator_y,
                self.generator_g_optimizer, self.generator_f_optimizer,
                self.discriminator_x_optimizer, self.discriminator_y_optimizer,
                image_x, image_y, self.loss_obj
            )
        return gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss

    def generate_and_visualize(self, epoch):
        # Generate and visualize images after each epoch
        sample_tr1 = next(iter(self.tr1_train))
        sample_tr2 = next(iter(self.tr2_train))
        generate_images(self.generator_g, sample_tr1,
                        self.generator_f, sample_tr2, epoch)
        
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

    def train(self):
        loader = DataLoader(val_dataset, batch_size, shuffle=False,
                            num_workers=12, pin_memory=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        mse_fn  = MeanSquaredError().to(device)
        psnr_scores, ssim_scores, mse_scores = [], [], []
        epochs_trained = 0

        # Train for the specified number of epochs
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = self.train_epoch()

            # Print the loss for the epoch
            print(
                f"Gen G Loss: {gen_g_loss}, Gen F Loss: {gen_f_loss}, Disc X Loss: {disc_x_loss}, Disc Y Loss: {disc_y_loss}")

            #Check Early Stopping
            val_loss = self._val_loss()
            print(f"  Val loss: {val_loss:.4f}")

            self.early_stopping(val_loss, self._models_dict())
            epochs_trained = epoch + 1

            # Visualize the results
            self.generate_and_visualize(epoch)

            # Calculate metrics on the test set
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating CycleGAN"):
                real_A = batch[DOMAIN_A_KEY]['input'].to(device)
                real_B = batch[DOMAIN_B_KEY]['input'].to(device)

                quant_t_a, quant_b_a, *_ = lit_model.domain_models[DOMAIN_A_KEY].encode(real_A)
                A_to_B = lit_model.domain_models[DOMAIN_B_KEY].decode(quant_t_a, quant_b_a)

                psnr_scores.append(psnr_fn(A_to_B, real_B).item())
                ssim_scores.append(ssim_fn(A_to_B, real_B).item())
                mse_scores.append(mse_fn(A_to_B,  real_B).item())

        print("\nTraining completed!")

        return {
            "model":          "cycle_GAN",
            "modality_1":     self.modality_1,
            "modality_2":     self.modality_2,
            "epochs_trained": epochs_trained,
            "ssim": float(np.mean(ssim_scores)),
            "psnr": float(np.mean(psnr_scores)),
            "mse":  float(np.mean(mse_scores)),
        }
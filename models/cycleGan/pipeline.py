import pathlib
import torch
from data_processing import load_datasets
from mri_gan import UNetGenerator, PatchDiscriminator
from training import train_step, evaluate_test_loss, evaluate_val_metrics
from early_stopping import EarlyStopping


class MRIGANPipeline:
    def __init__(self, training_data, testing_data,
                 modality_1, modality_2,
                 img_height, img_width, batch_size, buffer_size, epochs,
                 early_stopping_patience=5, early_stopping_delta=0.01):

        self.epochs     = epochs
        self.modality_1 = modality_1
        self.modality_2 = modality_2
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gen_g  = UNetGenerator().to(self.device)
        self.gen_f  = UNetGenerator().to(self.device)
        self.disc_x = PatchDiscriminator().to(self.device)
        self.disc_y = PatchDiscriminator().to(self.device)

        self.opt_g  = torch.optim.Adam(self.gen_g.parameters(),  lr=2e-4, betas=(0.5, 0.999))
        self.opt_f  = torch.optim.Adam(self.gen_f.parameters(),  lr=2e-4, betas=(0.5, 0.999))
        self.opt_dx = torch.optim.Adam(self.disc_x.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_dy = torch.optim.Adam(self.disc_y.parameters(), lr=2e-4, betas=(0.5, 0.999))

        (self.tr1_train, self.tr2_train,
         self.tr1_test,  self.tr2_test,
         self.tr1_val,   self.tr2_val) = load_datasets(
            img_height, img_width, batch_size, buffer_size)

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=early_stopping_delta)

    def _model_states(self):
        return {
            'gen_g':  self.gen_g.state_dict().copy(),
            'gen_f':  self.gen_f.state_dict().copy(),
            'disc_x': self.disc_x.state_dict().copy(),
            'disc_y': self.disc_y.state_dict().copy(),
        }

    def train(self):
        for epoch in range(1, self.epochs + 1):
            for real_x, real_y in zip(self.tr1_train, self.tr2_train):
                real_x = real_x.to(self.device)
                real_y = real_y.to(self.device)
                train_step(
                    self.gen_g, self.gen_f, self.disc_x, self.disc_y,
                    self.opt_g, self.opt_f, self.opt_dx, self.opt_dy,
                    real_x, real_y)

            test_loss = evaluate_test_loss(
                self.gen_g, self.gen_f,
                self.tr1_test, self.tr2_test, self.device)

            self.early_stopping(test_loss, self._model_states())

            if self.early_stopping.early_stop:
                break

        self.early_stopping.restore({
            'gen_g': self.gen_g, 'gen_f': self.gen_f,
            'disc_x': self.disc_x, 'disc_y': self.disc_y})

        metrics = evaluate_val_metrics(
            self.gen_g, self.gen_f,
            self.tr1_val, self.tr2_val, self.device)

        return {
            'model': 'CycleGAN',
            'mod1':  self.modality_1,
            'mod2':  self.modality_2,
            'ssim':  float(metrics['ssim']),
            'psnr':  float(metrics['psnr']),
            'mse':   float(metrics['mse']),
        }
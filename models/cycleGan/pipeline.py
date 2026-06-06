import torch
from torch.utils.data import DataLoader

from datasets import HDF5MultiDomainDataset
from mri_gan import UNetGenerator, PatchDiscriminator
from training import train_step, run_epoch, evaluate_test_loss_h5, evaluate_val_metrics_h5
from early_stopping import EarlyStopping

# Maps modality display names → HDF5 dataset key suffixes
# Must stay in sync with Clasp_2D_main.py / CKPT_TO_HDF5 there
CKPT_TO_HDF5 = {
    'Axial_T1_post': 't1ce_transformed',
    'Axial_T1_pre':  't1_transformed',
    'Axial_T2':      't2_transformed',
    'Axial_FLAIR':   'flair_transformed',
}


class MRIGANPipeline:
    """
    CycleGAN pipeline that trains on paired MRI slices loaded from HDF5 files.

    Parameters
    ----------
    train_files : list[Path | str]
        HDF5 files used for training.
    test_files : list[Path | str]
        HDF5 files used for early-stopping signal during training.
    val_files : list[Path | str]
        HDF5 files used for final metric evaluation (SSIM / PSNR / MSE)
        after the model is frozen.
    modality_1 : str
        Source domain name, e.g. 'Axial_T1_post'.
    modality_2 : str
        Target domain name, e.g. 'Axial_T2'.
    batch_size : int
    buffer_size : int
        Kept for API compatibility; DataLoader shuffle handles buffering.
    epochs : int
    early_stopping_patience : int
    early_stopping_delta : float
    img_height : int
        Not used by HDF5 loader (slices are returned at native resolution),
        kept so the caller signature matches comparison_pipeline.py.
    img_width : int
        Same as img_height.
    num_workers : int
        DataLoader worker processes (default 4).
    """

    def __init__(
        self,
        train_files,
        test_files,
        val_files,
        modality_1: str,
        modality_2: str,
        batch_size: int       = 32,
        buffer_size: int      = 1000,
        epochs: int           = 100,
        early_stopping_patience: int   = 5,
        early_stopping_delta:  float   = 0.01,
        img_height: int       = 256,
        img_width: int        = 256,
        num_workers: int      = 4,
    ):
        self.modality_1 = modality_1
        self.modality_2 = modality_2
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.es_patience = early_stopping_patience
        self.es_delta    = early_stopping_delta
        self.num_workers = num_workers

        # Resolve the HDF5 key suffixes for both domains
        self.domain_a_key = CKPT_TO_HDF5.get(modality_1)
        self.domain_b_key = CKPT_TO_HDF5.get(modality_2)

        if not self.domain_a_key or not self.domain_b_key:
            raise ValueError(
                f"Unknown modality: {modality_1!r} or {modality_2!r}. "
                f"Valid options: {list(CKPT_TO_HDF5.keys())}"
            )

        self.domain_key_prefixes = [self.domain_a_key, self.domain_b_key]

        # Build datasets from the three H5 file lists
        self.train_loader, self.test_loader, self.val_loader = \
            self._build_loaders(train_files, test_files, val_files)

        # GPU / CPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CycleGAN] Using device: {self.device}")

        # Instantiate models
        self.gen_g  = UNetGenerator().to(self.device)   # A → B
        self.gen_f  = UNetGenerator().to(self.device)   # B → A
        self.disc_x = PatchDiscriminator().to(self.device)
        self.disc_y = PatchDiscriminator().to(self.device)

        # Optimisers (Adam with standard CycleGAN hyperparameters)
        self.opt_g  = torch.optim.Adam(self.gen_g.parameters(),  lr=2e-4, betas=(0.5, 0.999))
        self.opt_f  = torch.optim.Adam(self.gen_f.parameters(),  lr=2e-4, betas=(0.5, 0.999))
        self.opt_dx = torch.optim.Adam(self.disc_x.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_dy = torch.optim.Adam(self.disc_y.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_loaders(self, train_files, test_files, val_files):
        """
        Build three DataLoaders from the respective H5 file lists.
        HDF5MultiDomainDataset handles prefix-matching and paired slice
        extraction for both modalities at once.
        """
        def make_loader(h5_files, shuffle: bool) -> DataLoader:
            ds = HDF5MultiDomainDataset(
                h5_file_list=h5_files,
                domain_key_prefixes=self.domain_key_prefixes,
                transform=None,          # default ToTensor inside dataset
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

        train_loader = make_loader(train_files, shuffle=True)
        test_loader  = make_loader(test_files,  shuffle=False)
        val_loader   = make_loader(val_files,   shuffle=False)

        print(
            f"[CycleGAN] Loaders built — "
            f"train: {len(train_loader.dataset)} slices, "
            f"test: {len(test_loader.dataset)} slices, "
            f"val: {len(val_loader.dataset)} slices"
        )
        return train_loader, test_loader, val_loader

    def _extract_domain_pair(self, batch):
        """
        Pull the two domain tensors out of the dict returned by
        HDF5MultiDomainDataset and move them to the current device.

        Each batch entry is: { domain_key: {'input': tensor, 'output': tensor} }
        We use the 'input' tensor for both domains.

        HDF5MultiDomainDataset stacks grayscale slices to 3-channel RGB,
        so ToTensor gives [B, 3, H, W]. UNetGenerator and PatchDiscriminator
        both expect 1-channel input, so we average the three identical
        channels back to [B, 1, H, W].
        """
        def to_grayscale(t):
            t = t.to(self.device)
            if t.shape[1] == 3:
                t = t.mean(dim=1, keepdim=True)   # [B,3,H,W] → [B,1,H,W]
            return t

        real_x = to_grayscale(batch[self.domain_a_key]['input'])
        real_y = to_grayscale(batch[self.domain_b_key]['input'])
        return real_x, real_y

    def _model_states(self) -> dict:
        """Return copies of all four model state dicts for early stopping."""
        return {
            'gen_g':  {k: v.clone() for k, v in self.gen_g.state_dict().items()},
            'gen_f':  {k: v.clone() for k, v in self.gen_f.state_dict().items()},
            'disc_x': {k: v.clone() for k, v in self.disc_x.state_dict().items()},
            'disc_y': {k: v.clone() for k, v in self.disc_y.state_dict().items()},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Run the full training loop with H5-backed data.

        Returns
        -------
        dict
            ``{'model': 'CycleGAN', 'mod1': ..., 'mod2': ...,
               'ssim': float, 'psnr': float, 'mse': float}``
            Compatible with the result format expected by
            comparison_pipeline.py → plot_comparison_table().
        """
        early_stopping = EarlyStopping(
            patience=self.es_patience,
            delta=self.es_delta,
        )

        for epoch in range(1, self.epochs + 1):

            # ---- Training pass (tqdm bar lives inside run_epoch) ------
            run_epoch(
                self.gen_g, self.gen_f, self.disc_x, self.disc_y,
                self.opt_g, self.opt_f, self.opt_dx, self.opt_dy,
                self.train_loader,
                epoch=epoch,
                total_epochs=self.epochs,
                extract_fn=self._extract_domain_pair,
            )

            # ---- Early-stopping check (tqdm bar lives inside evaluate_test_loss_h5) ----
            test_loss = evaluate_test_loss_h5(
                self.gen_g, self.gen_f,
                self.test_loader,
                self._extract_domain_pair,
            )
            improved = early_stopping(test_loss, self._model_states())

            print(
                f"[CycleGAN] Epoch {epoch}/{self.epochs}  "
                f"test_cycle_loss={test_loss:.4f}  "
                f"{'✓ improved' if improved else f'no improvement ({early_stopping.counter}/{self.es_patience})'}"
            )

            if early_stopping.early_stop:
                print(f"[CycleGAN] Early stopping triggered at epoch {epoch}.")
                break

        # ---- Restore best weights ------------------------------------
        early_stopping.restore({
            'gen_g':  self.gen_g,
            'gen_f':  self.gen_f,
            'disc_x': self.disc_x,
            'disc_y': self.disc_y,
        })

        # ---- Final evaluation (tqdm bar lives inside evaluate_val_metrics_h5) ----
        metrics = evaluate_val_metrics_h5(
            self.gen_g, self.gen_f,
            self.val_loader,
            self._extract_domain_pair,
            self.device,
        )

        print(
            f"[CycleGAN] Val metrics — "
            f"SSIM={metrics['ssim']:.4f}  "
            f"PSNR={metrics['psnr']:.2f} dB  "
            f"MSE={metrics['mse']:.6f}"
        )

        return {
            'model': 'CycleGAN',
            'mod1':  self.modality_1,
            'mod2':  self.modality_2,
            'ssim':  float(metrics['ssim']),
            'psnr':  float(metrics['psnr']),
            'mse':   float(metrics['mse']),
        }

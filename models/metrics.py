import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError
from tqdm import tqdm
import tensorflow as tf


class compute_metrics:
    def __init__(self, generator_g=None, device=None):
        self.generator_g = generator_g
        self.METRIC_DEVICE = device if device is not None else torch.device('cpu')

        self.psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(self.METRIC_DEVICE)
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.METRIC_DEVICE)
        self.mse_fn  = MeanSquaredError().to(self.METRIC_DEVICE)

    def _compute_metrics_tensorflow(self, dataset_x, dataset_y, model):
        ssim_scores, psnr_scores, mse_scores = [], [], []

        dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
        for real_x, real_y in tqdm(dataset, desc=f"  Evaluating {model}", unit="batch"):
            fake_y = self.generator_g(real_x, training=False)

            pred  = torch.from_numpy(fake_y.numpy()).permute(0, 3, 1, 2).to(self.METRIC_DEVICE)
            truth = torch.from_numpy(real_y.numpy()).permute(0, 3, 1, 2).to(self.METRIC_DEVICE)

            with torch.no_grad():
                psnr_scores.append(self.psnr_fn(pred, truth).item())
                ssim_scores.append(self.ssim_fn(pred, truth).item())
                mse_scores.append(self.mse_fn(pred.contiguous(), truth.contiguous()).item())
        return (
            ssim_scores if ssim_scores else [0.0],
            psnr_scores if psnr_scores else [0.0],
            mse_scores  if mse_scores  else [0.0],
        )

    def _compute_metrics_pytorch(self, loader, step_fn, model_name="model"):
        """
        Generic PyTorch evaluation loop.

        loader:     any iterable yielding batches (DataLoader, etc.)
        step_fn:    callable(batch) -> (prediction, ground_truth) as torch tensors,
                    already whatever shape/range torchmetrics expects (data_range=2.0,
                    i.e. values in [-1, 1]). Do all model-specific unpacking,
                    forward passes, encode/decode, etc. inside this closure.
        model_name: string shown in the tqdm progress bar

        Returns (ssim_scores, psnr_scores, mse_scores) as lists of per-batch floats.
        """
        psnr_scores, ssim_scores, mse_scores = [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Evaluating {model_name}", unit="batch"):
                pred, truth = step_fn(batch)
                pred  = pred.to(self.METRIC_DEVICE)
                truth = truth.to(self.METRIC_DEVICE)

                psnr_scores.append(self.psnr_fn(pred, truth).item())
                ssim_scores.append(self.ssim_fn(pred, truth).item())
                mse_scores.append(self.mse_fn(pred, truth).item())

        return (
            ssim_scores if ssim_scores else [0.0],
            psnr_scores if psnr_scores else [0.0],
            mse_scores  if mse_scores  else [0.0],
        )
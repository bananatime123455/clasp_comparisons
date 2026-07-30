
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError
from tqdm import tqdm
import tensorflow as tf

class compute_metrics:
    def __init__(self, generator_g):
        self.generator_g = generator_g
        self.METRIC_DEVICE = torch.device('cpu')
        
        self.psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(self.METRIC_DEVICE)
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.METRIC_DEVICE)
        self.mse_fn  = MeanSquaredError().to(self.METRIC_DEVICE)
    
    def _compute_metrics(self, dataset_x, dataset_y):

        ssim_scores, psnr_scores, mse_scores = [], [], []

        dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
        for real_x, real_y in tqdm(dataset, desc="  Evaluating CycleGAN", unit="batch"):
            fake_y = self.generator_g(real_x, training=False)

            # TF NHWC -> torch NCHW, values left untouched
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
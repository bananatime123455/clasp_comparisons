import os
import torch
import glob
import torchvision
from lightning import LightningModule
from lightning.pytorch.loggers import MLFlowLogger
import torch.nn.functional as F
from termcolor import colored
from torch import nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Dict, Any
import time
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError


class LitVQVAE2DualDomainCrossover(LightningModule):
    def __init__(self,
                 domain_models: dict,
                 lr: float,
                 gamma: float,
                 image_size: tuple,
                 recon_loss_fn: nn.Module,
                 crossover_loss_fn: nn.Module,
                 log_ids_every_n_steps: int = 100,
                 lambda_cross: float = 10,
                 lambda_recon: float = 1,
                 num_vis_val_batches=4,
                 num_vis_val_slices_per_batch=4,
                 log_train_images_every_n_epochs=10,
                 num_vis_train_slices=8
                 ):
        """
        Initializes the Lit VQVAE2 Crossover LightningModule for multi-domain training.

        Args:
            domain_models (dict): A dictionary mapping domain keys to their respective model instances.
            lr (float): Learning rate.
            gamma (float): The weight for the commitment loss.
            image_size (tuple): The height/width of the input images.
        """
        super().__init__()
        #self.save_hyperparameters()
        self.save_hyperparameters(ignore=['domain_models', 'recon_loss_fn', 'crossover_loss_fn'])

        # Directly use the provided dictionary
        self.domain_models = nn.ModuleDict(domain_models)

        # Instantiate the loss functions
        self.recon_loss_fn = recon_loss_fn
        self.crossover_loss_fn = crossover_loss_fn

        # Get codebook info from the first model, assuming they are shared
        first_model = next(iter(self.domain_models.values()))
        self.n_embed_t = first_model.quantize_t.n_embed
        self.n_embed_b = first_model.quantize_b.n_embed

        # Accumulators for tracking codebook IDs
        self.training_code_counts_t = None
        self.training_code_counts_b = None
        self.validation_code_counts_t = None
        self.validation_code_counts_b = None
        self._last_val_batch = None

        # Accumulators for tracking validation step losses
        #self._validation_step_losses = []

        self.middle_val_batch_idx = 0

        # --- Validation Visualization State ---
        self.num_vis_val_batches = num_vis_val_batches
        self.num_vis_val_slices_per_batch = num_vis_val_slices_per_batch
        self.vis_val_batch_indices = []
        self.captured_val_batches = []

        # --- Training Visualization State ---
        self.log_train_images_every_n_epochs = log_train_images_every_n_epochs
        self.num_vis_train_slices = num_vis_train_slices
        self.captured_train_batch = None  # To hold the first batch of a training epoch

        # Metrics to measure Reconstruction and Crossover Quality
        data_range = 1.0  # Assuming images are normalized to [0, 1]

        # Validation Metrics
        self.val_psnr = PeakSignalNoiseRatio(data_range=data_range)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        self.val_mse  = MeanSquaredError()

        # Training Metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=data_range)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        #self.train_mse = MeanSquaredError()

        # A set to keep track of checkpoints that have already been logged to MLflow
        self.logged_checkpoints = set()

 
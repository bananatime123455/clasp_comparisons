import os
import sys
import torch
import h5py
import types
from termcolor import colored
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS


from torch.utils.data import DataLoader

from torch.utils.data import DataLoader
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

from vqvae2_crossover import Quantize, Encoder, Decoder, VQVAE2_Crossover
from custom_losses import CustomAggregateLoss 

from scipy import stats as st
from pathlib import Path

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

import torch.serialization
from omegaconf import DictConfig, ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig])


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

 

def unet_generator():
    # Update the project root and add to path
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    #constants
    RANDOM_SEED = 42 # from config
    TEST_SPLIT_RATIO = 0.2

    #Define the 2 domains for the CycleGAN
    DOMAIN_A_KEY = 'Axial_T1_pre'
    DOMAIN_B_KEY = 'Axial_T2'
    domain_key_prefixes = [DOMAIN_A_KEY, DOMAIN_B_KEY]

    #Paths
    MODEL_PATH = "/mnt/c/Users/majun/OneDrive/Desktop/UW Mad Research/cycleGAN/cycleGAN/model"
    CHECKPOINT_PATH = "/mnt/c/Users/majun/OneDrive/Desktop/UW Mad Research/cycleGAN/cycleGAN/model/t1pre_and_t2.ckpt"


    #model
    # This is a way to make this as flexible as possible and minimize code changes that could lead to insideous errors

    # We're gonna mimic loading a Hydra config file without actually loading it
    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()

    cfg.model.top_codebook_embedding_dim = 128
    cfg.model.top_codebook_num_embeddings = 1024
    cfg.model.bottom_codebook_embedding_dim = 128
    cfg.model.bottom_codebook_num_embeddings = 1024
    cfg.model.base_channels = 128
    cfg.model.decay_rate = 0.99
    cfg.model.input_channels = 3
    cfg.model.num_res_blocks = 2
    cfg.model.num_res_channels = 32

    ############################################################################
    # Instantiate the Model
    ############################################################################

    # ---------------------------------------------------------------------------
    # VQVAE2 Crossover Model (Dual Domain with Crossover)

    # --- 1. Instantiate the SHARED Codebooks ---
    # These are instantiated ONLY ONCE.
    # They will be passed as objects to both model_A and model_B.
    shared_quantize_t = Quantize(dim=cfg.model.top_codebook_embedding_dim,
                                n_embed=cfg.model.top_codebook_num_embeddings,
                                decay=cfg.model.decay_rate)

    shared_quantize_b = Quantize(dim=cfg.model.bottom_codebook_embedding_dim,
                                n_embed=cfg.model.bottom_codebook_num_embeddings,
                                decay=cfg.model.decay_rate)

    # --- 2. Instantiate Encoders and Decoders for Domain A ---
    # These are specific to Domain A
    enc_b_A = Encoder(in_channel=cfg.model.input_channels,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=4)

    enc_t_A = Encoder(in_channel=cfg.model.base_channels,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=2)

    dec_t_A = Decoder(in_channel=cfg.model.top_codebook_embedding_dim,
                    out_channel=cfg.model.top_codebook_embedding_dim,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=2)

    dec_A = Decoder(in_channel=cfg.model.top_codebook_embedding_dim + cfg.model.bottom_codebook_embedding_dim,
                    out_channel=cfg.model.input_channels,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=4)

    # --- 3. Instantiate Encoders and Decoders for Domain B ---
    # These are specific to Domain B's
    enc_b_B = Encoder(in_channel=cfg.model.input_channels,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=4)

    enc_t_B = Encoder(in_channel=cfg.model.base_channels,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=2)

    dec_t_B = Decoder(in_channel=cfg.model.top_codebook_embedding_dim,
                    out_channel=cfg.model.top_codebook_embedding_dim,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=2)

    dec_B = Decoder(in_channel=cfg.model.top_codebook_embedding_dim + cfg.model.bottom_codebook_embedding_dim,
                    out_channel=cfg.model.input_channels,
                    channel=cfg.model.base_channels,
                    n_res_block=cfg.model.num_res_blocks,
                    n_res_channel=cfg.model.num_res_channels,
                    stride=4)

    # --- 4. Instantiate the TWO VQVAE2_Crossover models, passing all required parameters ---
    # Model for Domain A
    model_A = VQVAE2_Crossover(
        enc_t=enc_t_A,
        enc_b=enc_b_A,
        dec_t=dec_t_A,
        dec=dec_A,
        quantize_t=shared_quantize_t,  # SHARED codebook
        quantize_b=shared_quantize_b,  # SHARED codebook
        in_channel=cfg.model.input_channels,  # Pass this as per your __init__
        channel=cfg.model.base_channels,  # Pass this as per your __init__
        n_res_block=cfg.model.num_res_blocks,  # Pass this as per your __init__
        n_res_channel=cfg.model.num_res_channels,  # Pass this as per your __init__
        decay=cfg.model.decay_rate  # Pass this as per your __init__
    )

    # Model for Domain B
    model_B = VQVAE2_Crossover(
        enc_t=enc_t_B,
        enc_b=enc_b_B,
        dec_t=dec_t_B,
        dec=dec_B,
        quantize_t=shared_quantize_t,  # SHARED codebook (same object as model_A uses)
        quantize_b=shared_quantize_b,  # SHARED codebook (same object as model_A uses)
        in_channel=cfg.model.input_channels,  # Pass this (same for both domains) as per your __init__
        channel=cfg.model.base_channels,  # Pass this as per your __init__
        n_res_block=cfg.model.num_res_blocks,  # Pass this as per your __init__
        n_res_channel=cfg.model.num_res_channels,  # Pass this as per your __init__
        decay=cfg.model.decay_rate  # Pass this as per your __init__
    )

    domain_models = {
        DOMAIN_A_KEY: model_A,
        DOMAIN_B_KEY: model_B
    }

    recon_loss_fn = CustomAggregateLoss(
        losses={
            "l1": nn.L1Loss(),
            "lpips": LPIPS(net_type="vgg", normalize=True),
        },
        weights={"l1": 1.0, "lpips": 0.80},
    )

    crossover_loss_fn = CustomAggregateLoss(
        losses={
            "l1": nn.L1Loss(),
            "lpips": LPIPS(net_type="vgg", normalize=True),
        },
        weights={"l1": 1.0, "lpips": 0.80},
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        lit_model = LitVQVAE2DualDomainCrossover.load_from_checkpoint(
            CHECKPOINT_PATH,
            domain_models=domain_models,
            recon_loss_fn=recon_loss_fn,
            crossover_loss_fn=crossover_loss_fn,
            weights_only=False
        )
        print("Model loaded successfully from checkpoint.")
        return lit_model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
            # no instance norm on first block, matching downsample(64, 4, False)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # ZeroPadding2D + Conv2D(512, 4, strides=1)
        self.conv = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # ZeroPadding2D + Conv2D(1, 4, strides=1)
        self.last = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.conv(x)
        return self.last(x)

def discriminator():
    return Discriminator()
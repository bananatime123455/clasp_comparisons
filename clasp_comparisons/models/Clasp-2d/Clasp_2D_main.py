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

from scipy import stats as st
from pathlib import Path

# Update the project root and add to path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vqvae2_crossover import Quantize, Encoder, Decoder, VQVAE2_Crossover
from datasets import HDF5MultiDomainDataset
from custom_losses import CustomAggregateLoss 


def Clasp_2D_model(mod1_val_path, mod2_val_path, mod1, mod2, results_dir):
    DOMAIN_A_KEY = mod1
    DOMAIN_B_KEY = mod2
    domain_key_prefixes = [DOMAIN_A_KEY, DOMAIN_B_KEY]

    checkpoint = os.path.join()
    CHECKPOINT_PATH = "/mnt/ISNLGroup/Mike_Workspace/Experiments/CVPR/model_checkpoints/t2_and_flair.ckpt"

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
            domain_models=domain_models, # TEMPORARY! This is only needed for older checkpoints that didn't save domain_models
            recon_loss_fn = recon_loss_fn,
            crossover_loss_fn = crossover_loss_fn
        )
        print("Model loaded successfully from checkpoint.")
    except Exception as e:
        print(colored(f"Failed to load model from checkpoint: {e}", 'red'))
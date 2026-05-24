import types
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.regression import MeanSquaredError

from vqvae2_crossover import Quantize, Encoder, Decoder, VQVAE2_Crossover
from datasets import HDF5MultiDomainDataset
from custom_losses import CustomAggregateLoss
from lit_model import LitVQVAE2DualDomainCrossover


def Clasp_2D_model(val_files, mod1, mod2, checkpoint_path):
    DOMAIN_A_KEY = mod1
    DOMAIN_B_KEY = mod2
    domain_key_prefixes = [DOMAIN_A_KEY, DOMAIN_B_KEY]

    cfg = types.SimpleNamespace()
    cfg.model = types.SimpleNamespace()
    cfg.model.top_codebook_embedding_dim     = 128
    cfg.model.top_codebook_num_embeddings    = 1024
    cfg.model.bottom_codebook_embedding_dim  = 128
    cfg.model.bottom_codebook_num_embeddings = 1024
    cfg.model.base_channels                  = 128
    cfg.model.decay_rate                     = 0.99
    cfg.model.input_channels                 = 3
    cfg.model.num_res_blocks                 = 2
    cfg.model.num_res_channels               = 32

    shared_quantize_t = Quantize(
        dim=cfg.model.top_codebook_embedding_dim,
        n_embed=cfg.model.top_codebook_num_embeddings,
        decay=cfg.model.decay_rate)

    shared_quantize_b = Quantize(
        dim=cfg.model.bottom_codebook_embedding_dim,
        n_embed=cfg.model.bottom_codebook_num_embeddings,
        decay=cfg.model.decay_rate)

    def make_encoder(stride):
        return Encoder(
            in_channel=cfg.model.input_channels if stride == 4 else cfg.model.base_channels,
            channel=cfg.model.base_channels,
            n_res_block=cfg.model.num_res_blocks,
            n_res_channel=cfg.model.num_res_channels,
            stride=stride)

    def make_dec_t():
        return Decoder(
            in_channel=cfg.model.top_codebook_embedding_dim,
            out_channel=cfg.model.top_codebook_embedding_dim,
            channel=cfg.model.base_channels,
            n_res_block=cfg.model.num_res_blocks,
            n_res_channel=cfg.model.num_res_channels,
            stride=2)

    def make_dec():
        return Decoder(
            in_channel=cfg.model.top_codebook_embedding_dim + cfg.model.bottom_codebook_embedding_dim,
            out_channel=cfg.model.input_channels,
            channel=cfg.model.base_channels,
            n_res_block=cfg.model.num_res_blocks,
            n_res_channel=cfg.model.num_res_channels,
            stride=4)

    def make_vqvae():
        return VQVAE2_Crossover(
            enc_t=make_encoder(stride=2),
            enc_b=make_encoder(stride=4),
            dec_t=make_dec_t(),
            dec=make_dec(),
            quantize_t=shared_quantize_t,
            quantize_b=shared_quantize_b,
            in_channel=cfg.model.input_channels,
            channel=cfg.model.base_channels,
            n_res_block=cfg.model.num_res_blocks,
            n_res_channel=cfg.model.num_res_channels,
            decay=cfg.model.decay_rate)

    domain_models = {
        DOMAIN_A_KEY: make_vqvae(),
        DOMAIN_B_KEY: make_vqvae(),
    }

    recon_loss_fn = CustomAggregateLoss(
        losses={"l1": nn.L1Loss(), "lpips": LPIPS(net_type="vgg", normalize=True)},
        weights={"l1": 1.0, "lpips": 0.80})

    crossover_loss_fn = CustomAggregateLoss(
        losses={"l1": nn.L1Loss(), "lpips": LPIPS(net_type="vgg", normalize=True)},
        weights={"l1": 1.0, "lpips": 0.80})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        lit_model = LitVQVAE2DualDomainCrossover.load_from_checkpoint(
            checkpoint_path,
            domain_models=domain_models,
            recon_loss_fn=recon_loss_fn,
            crossover_loss_fn=crossover_loss_fn)
    except Exception as e:
        print((f"Failed to load checkpoint: {e}"))
        return None

    lit_model.eval()
    lit_model.freeze()
    lit_model.to(device)

    val_dataset = HDF5MultiDomainDataset(
        h5_file_list=val_files,
        domain_key_prefixes=domain_key_prefixes,
        transform=None)

    loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=12, pin_memory=True)

    psnr_fn = PeakSignalNoiseRatio().to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse_fn  = MeanSquaredError().to(device)

    psnr_scores, ssim_scores, mse_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating CLASP-2D"):
            real_A = batch[DOMAIN_A_KEY].to(device)
            real_B = batch[DOMAIN_B_KEY].to(device)

            fake_B = lit_model.domain_models[DOMAIN_B_KEY].decode(
                        *lit_model.domain_models[DOMAIN_A_KEY].encode(real_A))

            fake_B = fake_B.clamp(0, 1)
            real_B = real_B.clamp(0, 1)

            psnr_scores.append(psnr_fn(fake_B, real_B).item())
            ssim_scores.append(ssim_fn(fake_B, real_B).item())
            mse_scores.append( mse_fn(fake_B,  real_B).item())

    return {
        'model': 'Clasp_2D',
        'mod1':  mod1,
        'mod2':  mod2,
        'ssim':  float(np.mean(ssim_scores)),
        'psnr':  float(np.mean(psnr_scores)),
        'mse':   float(np.mean(mse_scores)),
    }
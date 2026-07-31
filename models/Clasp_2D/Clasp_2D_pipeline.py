import types
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.regression import MeanSquaredError

from vqvae2_crossover import Quantize, Encoder, Decoder, VQVAE2_Crossover
from datasets import HDF5MultiDomainDataset
from custom_losses import CustomAggregateLoss
from lit_model import LitVQVAE2DualDomainCrossover
from models.metrics import compute_metrics

# Maps checkpoint domain keys -> HDF5 suffix keys
CKPT_TO_HDF5 = {
    'Axial_T1_post': 't1ce_transformed',
    'Axial_T1_pre':  't1_transformed',
    'Axial_T2':      't2_transformed',
    'Axial_FLAIR':   'flair_transformed',
}

def Clasp_2D_model(val_files, mod1, mod2, checkpoint_path, batch_size):
    CKPT_A_KEY = mod1  # e.g. 'Axial_T1_post' — matches checkpoint state dict
    CKPT_B_KEY = mod2  # e.g. 'Axial_T2'

    DOMAIN_A_KEY = CKPT_TO_HDF5.get(mod1)  # e.g. 't1ce_transformed' — matches HDF5
    DOMAIN_B_KEY = CKPT_TO_HDF5.get(mod2)  # e.g. 't2_transformed'

    if not DOMAIN_A_KEY or not DOMAIN_B_KEY:
        print(f"No HDF5 key mapping for {mod1!r} or {mod2!r}. "
              f"Valid options: {list(CKPT_TO_HDF5.keys())}")
        return None

    domain_key_prefixes = [DOMAIN_A_KEY, DOMAIN_B_KEY]

    #Load pseudo hydra config 

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
        CKPT_A_KEY: model_A,
        CKPT_B_KEY: model_B
    }



    recon_loss_fn = CustomAggregateLoss(
        losses={"l1": nn.L1Loss(), "lpips": LPIPS(net_type="vgg", normalize=True)},
        weights={"l1": 1.0, "lpips": 0.80})

    crossover_loss_fn = CustomAggregateLoss(
        losses={"l1": nn.L1Loss(), "lpips": LPIPS(net_type="vgg", normalize=True)},
        weights={"l1": 1.0, "lpips": 0.80})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        lit_model = LitVQVAE2DualDomainCrossover.load_from_checkpoint(
            checkpoint_path,
            weights_only=False,
            domain_models=domain_models,
            recon_loss_fn=recon_loss_fn,
            crossover_loss_fn=crossover_loss_fn)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

    lit_model.eval()
    lit_model.freeze()
    lit_model.to(device)

    # Alias checkpoint keys -> HDF5 keys for dataset batch lookups
    lit_model.domain_models[DOMAIN_A_KEY] = lit_model.domain_models[CKPT_A_KEY]
    lit_model.domain_models[DOMAIN_B_KEY] = lit_model.domain_models[CKPT_B_KEY]

    val_dataset = HDF5MultiDomainDataset(
        h5_file_list=val_files,
        domain_key_prefixes=domain_key_prefixes,
        transform=None)

    loader = DataLoader(val_dataset, batch_size, shuffle=False,
                        num_workers=12, pin_memory=True)

    metrics_computer = compute_metrics(device=device)

    def clasp_step(batch):
        real_A = batch[DOMAIN_A_KEY]['input'].to(device)
        real_B = batch[DOMAIN_B_KEY]['input'].to(device)
        quant_t_a, quant_b_a, *_ = lit_model.domain_models[DOMAIN_A_KEY].encode(real_A)
        A_to_B = lit_model.domain_models[DOMAIN_B_KEY].decode(quant_t_a, quant_b_a)
        return A_to_B, real_B

    ssim_scores, psnr_scores, mse_scores = metrics_computer._compute_metrics_pytorch(
        loader=loader,
        step_fn=clasp_step,
        model_name="CLASP-2D",
    )

    return {
        'model': 'Clasp_2D Cross domain',
        'mod1':  mod1,
        'mod2':  mod2,
        'ssim':  float(np.mean(ssim_scores)),
        'psnr':  float(np.mean(psnr_scores)),
        'mse':   float(np.mean(mse_scores)),
    }
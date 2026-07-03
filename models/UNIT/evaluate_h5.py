# models/UNIT/evaluate_h5.py
#
# Standalone evaluation script — loads a trained UNIT checkpoint, runs the
# generator on a given H5 file list, and computes SSIM/PSNR/MSE of the
# translated (modality_1 -> modality_2) output against real modality_2.
#
# Bypasses imaginaire's inference.py / net_G.inference() entirely — uses the
# same net_G_output['images_ab'] path that Trainer._get_visualizations() uses.
import sys
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.trainer import get_model_optimizer_and_scheduler, get_trainer, set_random_seed
from imaginaire.utils.distributed import init_dist
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.datasets.h5_mri import H5MRIDataset
from get_h5_dataloaders import _parse_aug

from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(description="UNIT H5 evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--file_list", required=True,
                        help="Path to .txt file of H5 paths to evaluate on")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)

    cfg = Config(args.config)
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)

    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Build eval dataset directly from the given file list.
    from pathlib import Path
    eval_files = [Path(p) for p in Path(args.file_list).read_text().splitlines() if p.strip()]
    eval_aug = _parse_aug(cfg.data.val.augmentations)  # reuse val-style center crop, no augmentation randomness
    eval_dataset = H5MRIDataset(
        h5_files=eval_files,
        modality_a=cfg.data.modality_a,
        modality_b=cfg.data.modality_b,
        augmentations=eval_aug,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Build model + load checkpoint (mirrors train.py/inference.py setup).
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, None, eval_loader)
    trainer.load_checkpoint(cfg, args.checkpoint)

    if cfg.trainer.model_average_config.enabled:
        net_G_eval = trainer.net_G.module.averaged_model
    else:
        net_G_eval = trainer.net_G.module
    net_G_eval.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    ssim_vals, psnr_vals, mse_vals = [], [], []

    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            data = trainer.start_of_iteration(data, current_iteration=-1)
            net_G_output = net_G_eval(data)

            # A -> B translation vs. real B (matches modality_1 -> modality_2)
            fake_b = net_G_output['images_ab']  # (B, C, H, W), range [-1, 1]
            real_b = data['images_b']

            fake_np = fake_b.clamp(-1, 1).cpu().numpy()
            real_np = real_b.clamp(-1, 1).cpu().numpy()

            for b in range(fake_np.shape[0]):
                # Convert (C, H, W) -> (H, W) grayscale, rescale [-1,1] -> [0,1]
                pred = (fake_np[b, 0] + 1) / 2
                truth = (real_np[b, 0] + 1) / 2

                mse_vals.append(float(mean_squared_error(truth, pred)))
                ssim_vals.append(float(structural_similarity(truth, pred, data_range=1.0)))
                psnr_vals.append(float(peak_signal_noise_ratio(truth, pred, data_range=1.0)))

            if i % 20 == 0:
                print(f"[evaluate_h5] {i}/{len(eval_loader)} samples done")

    metrics = {"ssim": ssim_vals, "psnr": psnr_vals, "mse": mse_vals}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"[evaluate_h5] {len(ssim_vals)} slices — "
          f"SSIM={np.mean(ssim_vals):.4f}  "
          f"PSNR={np.mean(psnr_vals):.2f} dB  "
          f"MSE={np.mean(mse_vals):.5f}")
    print(f"[evaluate_h5] Metrics written → {metrics_path}")


if __name__ == "__main__":
    main()
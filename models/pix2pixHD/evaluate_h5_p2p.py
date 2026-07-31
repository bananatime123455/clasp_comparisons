import sys, os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.metrics import compute_metrics

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.trainer import get_model_optimizer_and_scheduler, get_trainer, set_random_seed
from imaginaire.utils.distributed import init_dist
from imaginaire.utils.gpu_affinity import set_affinity
from h5_pix2pix_dataset import H5Pix2PixDataset
from get_h5_dataloaders import _parse_aug


def parse_args():
    p = argparse.ArgumentParser(description="pix2pixHD H5 evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--file_list", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--single_gpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)

    cfg = Config(args.config)
    if not args.single_gpu:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    from pathlib import Path
    eval_files = [Path(p) for p in Path(args.file_list).read_text().splitlines() if p.strip()]
    eval_aug = _parse_aug(cfg.data.val.augmentations)
    eval_dataset = H5Pix2PixDataset(eval_files, cfg.data.modality_a, cfg.data.modality_b, eval_aug)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    net_G, net_D, opt_G, opt_D, sch_G, sch_D = get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D, None, eval_loader)
    trainer.load_checkpoint(cfg, args.checkpoint)

    net_G_eval = (trainer.net_G.module.averaged_model
                  if cfg.trainer.model_average_config.enabled
                  else trainer.net_G.module)
    net_G_eval.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.single_gpu else 'cpu')
    metrics_computer = compute_metrics(device=device)

    def pix2pixhd_step(data):
        data = trainer.start_of_iteration(data, current_iteration=-1)
        net_G_output = net_G_eval(data)
        fake = net_G_output['fake_images'].clamp(-1, 1)   # <- confirmed key
        real = data['images'].clamp(-1, 1)
        return fake, real

    ssim_vals, psnr_vals, mse_vals = metrics_computer._compute_metrics_pytorch(
        loader=eval_loader, step_fn=pix2pixhd_step, model_name="pix2pixHD",
    )

    metrics = {"ssim": ssim_vals, "psnr": psnr_vals, "mse": mse_vals}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"[evaluate_h5_p2p] SSIM={np.mean(ssim_vals):.4f}  "
          f"PSNR={np.mean(psnr_vals):.2f}  MSE={np.mean(mse_vals):.5f}")


if __name__ == "__main__":
    main()
import random
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_key(h5_file: h5py.File, modality_suffix: str) -> str:
    """Return the full H5 key whose name ends with modality_suffix."""
    keys = list(h5_file.keys())
    match = next((k for k in keys if k.endswith(modality_suffix)), None)
    if match is None:
        raise KeyError(
            f"No key ending with '{modality_suffix}' found. "
            f"Available keys: {keys}"
        )
    return match


def _normalize_volume(arr: np.ndarray) -> np.ndarray:
    """
    Percentile-clip then normalize a 3D MRI volume to [0, 1].
    Clipping at 1st/99th percentile removes outlier intensities common in MRI.
    Operates on the full volume so slice-to-slice intensity is consistent.
    """
    arr = arr.astype(np.float32)
    lo  = np.percentile(arr, 1)
    hi  = np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    rng = hi - lo
    if rng < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / rng   # → [0, 1]


def _to_tensor(slc: np.ndarray, normalize_to_neg1_pos1: bool = True) -> torch.Tensor:
    """
    2D float32 slice (H, W) → (3, H, W) float32 tensor.
    Channel is repeated 3× so UNIT's VGG perceptual path and discriminator
    architecture (which expects 3-channel input) work without modification.
    normalize_to_neg1_pos1: scales [0,1] → [-1,1] as UNIT expects.
    """
    t = torch.from_numpy(slc).float()      # (H, W)
    t = t.unsqueeze(0).repeat(3, 1, 1)    # (3, H, W)
    if normalize_to_neg1_pos1:
        t = t * 2.0 - 1.0                  # [0,1] → [-1,1]
    return t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class H5MRIDataset(Dataset):
    """
    Unpaired MRI dataset loaded purely from H5 files as numpy arrays.

    Builds a flat index of (h5_path, slice_index) at construction time,
    then loads individual slices on demand in __getitem__ so memory usage
    stays low even for large cohorts.

    Parameters
    ----------
    h5_files : list[Path]
        H5 files for this split (train / val / test).
    modality_a : str
        Key suffix for domain A, e.g. 't1ce_transformed'.
    modality_b : str
        Key suffix for domain B, e.g. 't2_transformed'.
    augmentations : dict | None
        Augmentation config from YAML — handled keys:
            resize_smallest_side  int
            random_scale_limit    float
            random_crop_h_w       tuple[int, int]
            center_crop_h_w       tuple[int, int]
            horizontal_flip       bool
    blank_threshold : float
        Slices where both modalities have mean < this (on [0,1] scale)
        are excluded from the index to avoid trivially easy blank inputs.
    """

    def __init__(
        self,
        h5_files: list,
        modality_a: str,
        modality_b: str,
        augmentations: Optional[dict] = None,
        blank_threshold: float = 0.05,
    ):
        self.modality_a      = modality_a
        self.modality_b      = modality_b
        self.augmentations   = augmentations or {}
        self.blank_threshold = blank_threshold

        # Separate indices per domain — truly unpaired
        self.index_a: List[Tuple[Path, int]] = []
        self.index_b: List[Tuple[Path, int]] = []

        print(f"[H5MRIDataset] Indexing {len(h5_files)} files ...")
        for h5_path in h5_files:
            h5_path = Path(h5_path)
            with h5py.File(h5_path, "r") as f:
                key_a = _find_key(f, modality_a)
                key_b = _find_key(f, modality_b)
                vol_a = _normalize_volume(np.array(f[key_a]))  # (S, H, W)
                vol_b = _normalize_volume(np.array(f[key_b]))

            n_slices = min(vol_a.shape[0], vol_b.shape[0])
            kept = 0
            for i in range(n_slices):
                if (vol_a[i].mean() < blank_threshold and
                        vol_b[i].mean() < blank_threshold):
                    continue
                self.index_a.append((h5_path, i))
                self.index_b.append((h5_path, i))
                kept += 1

            print(f"  {h5_path.name}: {kept}/{n_slices} slices indexed")

        print(
            f"[H5MRIDataset] Ready — "
            f"{len(self.index_a)} A-slices, {len(self.index_b)} B-slices"
        )

    def __len__(self) -> int:
        return max(len(self.index_a), len(self.index_b))

    def __getitem__(self, idx: int) -> dict:
        path_a, si_a = self.index_a[idx % len(self.index_a)]
        # Unpaired: random B slice — domains are never aligned
        path_b, si_b = random.choice(self.index_b)

        slc_a = self._load_slice(path_a, self.modality_a, si_a)
        slc_b = self._load_slice(path_b, self.modality_b, si_b)

        ta = _to_tensor(slc_a)
        tb = _to_tensor(slc_b)

        ta, tb = self._augment(ta, tb)

        return {"images_a": ta, "images_b": tb}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_slice(self, h5_path: Path, modality: str, slice_idx: int) -> np.ndarray:
        """Load and normalize a single slice on demand."""
        with h5py.File(h5_path, "r") as f:
            key = _find_key(f, modality)
            vol = _normalize_volume(np.array(f[key]))
        return vol[slice_idx]   # (H, W) float32 in [0, 1]

    def _augment(
        self,
        ta: torch.Tensor,
        tb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the same spatial augmentation to both domains so geometry
        stays consistent even though the images are unpaired content-wise.
        """
        aug = self.augmentations

        if "resize_smallest_side" in aug:
            size = int(aug["resize_smallest_side"])
            ta = TF.resize(ta, size, antialias=True)
            tb = TF.resize(tb, size, antialias=True)

        if "random_scale_limit" in aug:
            limit = float(aug["random_scale_limit"])
            scale = 1.0 + random.uniform(-limit, limit)
            h, w  = ta.shape[-2], ta.shape[-1]
            ta = TF.resize(ta, [int(h * scale), int(w * scale)], antialias=True)
            tb = TF.resize(tb, [int(h * scale), int(w * scale)], antialias=True)

        if "random_crop_h_w" in aug:
            ch, cw = aug["random_crop_h_w"]
            ch, cw = int(ch), int(cw)
            # Same crop coords for both domains
            i = random.randint(0, max(ta.shape[-2] - ch, 0))
            j = random.randint(0, max(ta.shape[-1] - cw, 0))
            ta = TF.crop(ta, i, j, ch, cw)
            tb = TF.crop(tb, i, j, ch, cw)
        elif "center_crop_h_w" in aug:
            ch, cw = aug["center_crop_h_w"]
            ta = TF.center_crop(ta, [int(ch), int(cw)])
            tb = TF.center_crop(tb, [int(ch), int(cw)])

        if aug.get("horizontal_flip", False) and random.random() > 0.5:
            ta = TF.hflip(ta)
            tb = TF.hflip(tb)

        return ta, tb
# models/UNIT/get_h5_dataloaders.py
#
# Drop-in replacement for imaginaire.utils.dataset.get_train_and_val_dataloader
# Feeds pure numpy arrays from H5 files — no PNG conversion.
#FOR UNIT AND MUNIT
# Called by pipeline.py — never needs to be touched directly.

from pathlib import Path
from torch.utils.data import DataLoader

from imaginaire.datasets.h5_mri import H5MRIDataset


def get_train_and_test_dataloader(cfg, train_files: list, test_files: list):
    """
    Build train and test DataLoaders directly from H5 file lists.

    Parameters
    ----------
    cfg : imaginaire Config object
        Parsed from the YAML — reads:
            cfg.data.modality_a
            cfg.data.modality_b
            cfg.data.num_workers
            cfg.data.train.batch_size
            cfg.data.test.batch_size
            cfg.data.train.augmentations
            cfg.data.test.augmentations
    train_files : list[Path]
        Already-split train H5 paths from sort_data().
    test_files : list[Path]
        Already-split val H5 paths from sort_data().
    """
    modality_a  = cfg.data.modality_a
    modality_b  = cfg.data.modality_b
    num_workers = getattr(cfg.data, "num_workers", 4)

    train_aug = _parse_aug(cfg.data.train.augmentations)
    test_aug   = _parse_aug(cfg.data.test.augmentations)

    print(f"[get_h5_dataloaders] train={len(train_files)} subjects, "
          f"test={len(test_files)} subjects")

    train_dataset = H5MRIDataset(
        h5_files=train_files,
        modality_a=modality_a,
        modality_b=modality_b,
        augmentations=train_aug,
        paired=False,   # UNIT/MUNIT train unpaired by design
    )
    test_dataset = H5MRIDataset(
        h5_files=test_files,
        modality_a=modality_a,
        modality_b=modality_b,
        augmentations=test_aug,
        # paired defaults to True — val/test loss should reflect true pairs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,             # consistent batch sizes for GAN training
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.val.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, test_loader


def _parse_aug(aug_cfg) -> dict:
    """
    Convert imaginaire's augmentation config object to a plain dict.
    Parses 'random_crop_h_w: 256, 256' style strings into int tuples.
    """
    if aug_cfg is None:
        return {}
    d = aug_cfg if isinstance(aug_cfg, dict) else (
        vars(aug_cfg) if hasattr(aug_cfg, "__dict__") else dict(aug_cfg)
    )
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and "," in v:
            try:
                v = tuple(int(p.strip()) for p in v.split(","))
            except ValueError:
                pass
        out[k] = v
    return out

def get_val_dataloader(cfg, val_files: list):
    """
    Build a DataLoader for evaluation, reusing H5MRIDataset the same way
    get_train_and_val_dataloader does. Used by inference_h5.py to feed
    validation H5 files through inference.py's unmodified test path.
    """
    modality_a = cfg.data.modality_a
    modality_b = cfg.data.modality_b
    num_workers = getattr(cfg.data, "num_workers", 4)

    test_aug = _parse_aug(cfg.test_data.test.augmentations)

    print(f"[get_h5_dataloaders] test={len(val_files)} subjects")

    test_dataset = H5MRIDataset(
        h5_files=val_files,
        modality_a=modality_a,
        modality_b=modality_b,
        augmentations=test_aug,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=getattr(cfg.test_data.test, "batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return val_loader
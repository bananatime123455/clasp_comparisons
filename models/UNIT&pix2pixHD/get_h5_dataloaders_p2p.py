from torch.utils.data import DataLoader
from h5_pix2pix_dataset import H5Pix2PixDataset
from get_h5_dataloaders import _parse_aug  # reuse the aug parser from UNIT's module


def get_train_and_test_dataloader(cfg, train_files: list, test_files: list):
    modality_a  = cfg.data.modality_a
    modality_b  = cfg.data.modality_b
    num_workers = getattr(cfg.data, "num_workers", 4)

    train_aug = _parse_aug(cfg.data.train.augmentations)
    test_aug  = _parse_aug(cfg.data.val.augmentations)

    train_dataset = H5Pix2PixDataset(train_files, modality_a, modality_b, train_aug)
    test_dataset  = H5Pix2PixDataset(test_files, modality_a, modality_b, test_aug)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.train.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.data.val.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
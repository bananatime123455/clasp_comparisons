from torch.utils.data import Dataset
from imaginaire.datasets.h5_mri import H5MRIDataset


class H5Pix2PixDataset(Dataset):
    """
    Wraps H5MRIDataset (built for UNIT's images_a/images_b keys) and remaps
    to the keys pix2pixHD's generator expects:
        images_a -> label    (modality_1, continuous — no one-hot)
        images_b -> images   (modality_2, the real target)
    """
    def __init__(self, h5_files, modality_a, modality_b, augmentations):
        self._base = H5MRIDataset(
            h5_files=h5_files,
            modality_a=modality_a,
            modality_b=modality_b,
            augmentations=augmentations,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        item = self._base[idx]
        out = {
            'label':  item['images_a'],
            'images': item['images_b'],
        }
        # Only needed if gen.concat_features / an encoder block is enabled
        # in the yaml. If you never set gen.enc, this key is likely unused —
        # remove it if imaginaire complains about an unexpected key.
        if 'instance_maps' not in out:
            out['instance_maps'] = item['images_a'][:1]  # 1-channel dummy
        return out
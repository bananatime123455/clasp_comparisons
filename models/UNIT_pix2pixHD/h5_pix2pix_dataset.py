# h5_pix2pix_dataset.py
from collections import OrderedDict
from torch.utils.data import Dataset
from imaginaire.datasets.h5_mri import H5MRIDataset


class H5Pix2PixDataset(Dataset):
    def __init__(self, h5_files, modality_a, modality_b, augmentations, label_channels=3):
        self._base = H5MRIDataset(
            h5_files=h5_files,
            modality_a=modality_a,
            modality_b=modality_b,
            augmentations=augmentations,
        )
        self._label_channels = label_channels  # matches yaml's `label: num_channels: 3`

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        item = self._base[idx]
        out = {
            'label':  item['images_a'],
            'images': item['images_b'],
        }
        if 'instance_maps' not in out:
            out['instance_maps'] = item['images_a'][:1]
        return out

    def get_label_lengths(self):
        """
        Tells imaginaire's visualizer how to split the concatenated `label`
        tensor back into named sub-fields for preview-image logging.
        Since our `label` is one continuous field (modality_1), not a
        one-hot seg_map + instance_map stack, there's only one entry.
        """
        return OrderedDict([('label', self._label_channels)])
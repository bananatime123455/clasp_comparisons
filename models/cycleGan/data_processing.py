import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sort_data import sort_data


class MRIDataset(Dataset):
    def __init__(self, paths, img_height=256, img_width=256):
        self.paths      = paths
        self.img_height = img_height
        self.img_width  = img_width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L').resize(
            (self.img_width, self.img_height), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # → [-1, 1]
        return torch.from_numpy(arr).unsqueeze(0)               # [1, H, W]


def load_datasets(data_dir_mod1, data_dir_mod2, img_height, img_width,
                  batch_size, num_workers=4):

    mod1_paths = sorted(pathlib.Path(data_dir_mod1).glob('**/*.png'))
    mod2_paths = sorted(pathlib.Path(data_dir_mod2).glob('**/*.png'))

    mod1_train, mod1_test, mod1_val = sort_data(mod1_paths)
    mod2_train, mod2_test, mod2_val = sort_data(mod2_paths)

    def make_loader(paths, shuffle):
        ds = MRIDataset(paths, img_height, img_width)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)

    return (
        make_loader(mod1_train, shuffle=False),
        make_loader(mod2_train, shuffle=False),
        make_loader(mod1_test,  shuffle=False),
        make_loader(mod2_test,  shuffle=False),
        make_loader(mod1_val,   shuffle=False),
        make_loader(mod2_val,   shuffle=False),
    )
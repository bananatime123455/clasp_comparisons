import sys
import os

#ensure local imports found first
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#ensure h5 data is loaded instead of LMDB/folder data
import imaginaire.utils.dataset as _iu_dataset
from get_h5_dataloaders import get_train_and_val_dataloader as _h5_loader
from imaginaire.config import Config as _Config


def _patched_loader(cfg, seed=0):
    """
    Reads the file lists written by pipeline._write_config() and passes
    them to our H5 dataloader instead of imaginaire's LMDB/folder loader.
    """
    from pathlib import Path

    train_list = Path(cfg.data.train_file_list)
    val_list   = Path(cfg.data.val_file_list)

    train_files = [Path(p) for p in train_list.read_text().splitlines() if p.strip()]
    val_files   = [Path(p) for p in val_list.read_text().splitlines()   if p.strip()]

    return _h5_loader(cfg, train_files=train_files, val_files=val_files)


#set it so that imaginaire's train.py will use our H5 dataloader instead of the default LMDB/folder loader
_iu_dataset.get_train_and_val_dataloader = _patched_loader


#start training by executing imaginaire's train.py, but with our patched dataloader
train_py = os.path.join(_HERE, "train.py")
with open(train_py) as f:
    _code = f.read()

exec(compile(_code, train_py, "exec"), {"__name__": "__main__", "__file__": train_py})

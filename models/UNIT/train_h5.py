# models/UNIT/train_h5.py
#
# Thin wrapper around imaginaire's train.py that replaces
# get_train_and_val_dataloader with our H5 numpy version.
#
# Called by pipeline.py inside Docker — accepts identical CLI args to train.py
# so no changes to the Docker command are needed beyond swapping the script name.
#
# The patch works by importing get_h5_dataloaders before imaginaire.utils.trainer
# touches the dataloader, then monkey-patching the symbol in imaginaire's
# utils.dataset module so every subsequent import resolves to our version.

import sys
import os

# ---------------------------------------------------------------------------
# 1. Ensure models/UNIT is on the path so our local modules are found first
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# 2. Patch imaginaire's dataloader BEFORE train.py imports it
# ---------------------------------------------------------------------------
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


# Monkey-patch so any code doing
#   from imaginaire.utils.dataset import get_train_and_val_dataloader
# after this point gets our version.
_iu_dataset.get_train_and_val_dataloader = _patched_loader

# ---------------------------------------------------------------------------
# 3. Now run train.py's main() — it will pick up the patched loader
# ---------------------------------------------------------------------------
# We exec train.py rather than importing it so __name__ == '__main__' works
# and argparse reads sys.argv correctly.
train_py = os.path.join(_HERE, "train.py")
with open(train_py) as f:
    _code = f.read()

exec(compile(_code, train_py, "exec"), {"__name__": "__main__", "__file__": train_py})

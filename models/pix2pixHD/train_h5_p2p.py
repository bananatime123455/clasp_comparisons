import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import imaginaire.utils.dataset as _iu_dataset
from get_h5_dataloaders_p2p import get_train_and_test_dataloader as _h5_loader


def _patched_loader(cfg, seed=0):
    from pathlib import Path
    train_files = [Path(p) for p in Path(cfg.data.train_file_list).read_text().splitlines() if p.strip()]
    test_files  = [Path(p) for p in Path(cfg.data.val_file_list).read_text().splitlines() if p.strip()]
    return _h5_loader(cfg, train_files=train_files, test_files=test_files)


_iu_dataset.get_train_and_val_dataloader = _patched_loader

train_py = os.path.join(_HERE, "train.py")  # reuse imaginaire's unmodified train.py
with open(train_py) as f:
    _code = f.read()
exec(compile(_code, train_py, "exec"), {"__name__": "__main__", "__file__": train_py})
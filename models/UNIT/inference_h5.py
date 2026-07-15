import sys
import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


#parse extra --file_list arg, leave the rest for inference.py

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--file_list", required=True,
                         help="Path to .txt file of H5 paths to evaluate on.")
_known, _remaining = _pre_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

from pathlib import Path
_eval_files = [Path(p) for p in Path(_known.file_list).read_text().splitlines() if p.strip()]

#ensure h5 data is loaded instead of LMDB/folder data
import imaginaire.utils.dataset as _iu_dataset
from get_h5_dataloaders import get_test_dataloader as _h5_test_loader

def _patched_test_loader(cfg):
    return _h5_test_loader(cfg, test_files=_eval_files)

_iu_dataset.get_test_dataloader = _patched_test_loader

#start inference by executing imaginaire's inference.py, but with our patched dataloader
infer_py = os.path.join(_HERE, "inference.py")
with open(infer_py) as f:
    _code = f.read()
exec(compile(_code, infer_py, "exec"), {"__name__": "__main__", "__file__": infer_py})
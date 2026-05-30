import h5py
from pathlib import Path

h5_path = "/mnt/r/User_Workspaces/Eric_Ma/New_Registration_Results/"
h5_files = sorted(Path(h5_path).glob("**/*.h5"))
prefixes = ['t1ce_transformed', 't2_transformed']

print(f"Found {len(h5_files)} files")
for h5_path in h5_files:
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print(f"\nFile: {h5_path}")
        print(f"Keys: {keys}")
        for prefix in prefixes:
            match = next((key for key in keys if key.endswith(prefix)), None)
            print(f"  prefix '{prefix}' -> matched: {match}")
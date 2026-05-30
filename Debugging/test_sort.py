from sort_data import sort_data
from pathlib import Path

h5_files = sorted(Path("/mnt/r/User_Workspaces/Eric_Ma/New_Registration_Results/").glob("**/*.h5"))
train, test, val = sort_data(h5_files)
print(f"train: {len(train)}")
print(f"test:  {len(test)}")
print(f"val:   {len(val)}")
print(f"val files: {val}")
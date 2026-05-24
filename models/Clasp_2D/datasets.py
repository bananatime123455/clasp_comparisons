import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import imageio.v3 as iio
from termcolor import colored
import torchvision.transforms as T

class GenericMRI2DImageDataset(Dataset):
    """
    Dataset for loading 2D MRI slices from TIFF, JPG, or PNG files.
    - Converts grayscale to RGB
    - Normalizes float TIFFs to [0, 1] (required by ToTensor())
    - Ensures all inputs are compatible with ToTensor()
    """

    def __init__(self, file_list, transform=None):
        """
        Args:
            file_list (List[str]): List of image file paths.
            transform (callable, optional): Transformations to apply (e.g., ToTensor, Normalize).
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        image = iio.imread(path)

        # If grayscale (H, W), convert to RGB (H, W, 3)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Ensure we're working with the same datatype
        image = image.astype(np.float32)

        # Min-Max Scale Images
        img_min = np.min(image)
        img_max = np.max(image)
        img_range = img_max-img_min

        image = (image - img_min) / (img_range + 1e-8)

        ''' Original normalization
        # Normalize depending on dtype
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Normalize float images to [0, 1]
            image = image / (np.max(image) + 1e-8) # Add some laplace smoothing
        else:
            # Assume uint8, scale from [0, 255]
            image = image.astype(np.float32) / 255.0

        '''
        # Apply transform (e.g., ToTensor, Alternative Normalization, Resize, etc.)
        if self.transform:
            image = self.transform(image)

        return image


class BaseHDF5Dataset(Dataset):
    """
    A base class for HDF5 datasets that provides shared preprocessing logic.
    Args:
            transform (callable, optional): PyTorch transforms.
    """
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

    @staticmethod
    def _preprocess_slice(image_slice, global_stats=None):
        # If grayscale (H, W), convert to RGB (H, W, 3)
        if image_slice.ndim == 2:
            image_slice = np.stack([image_slice] * 3, axis=-1)

        # Ensure we're working with the same datatype
        image_slice = image_slice.astype(np.float32)

        if global_stats is not None:
            # Use pre-computed volume stats
            img_min, img_max = global_stats
        else:
            # Fallback to slice-wise stats (Legacy MRI support)
            img_min = np.min(image_slice)
            img_max = np.max(image_slice)

        # Min-Max Scale Images
        img_range = img_max - img_min
        image_slice = (image_slice - img_min) / (img_range + 1e-8)

        return image_slice

class HDF52DSingleDomainDatasetV2(BaseHDF5Dataset):
    """
    Dataset for loading 2D MRI slices for a SINGLE DOMAIN from HDF5 files.
    - It maps a linear index to a specific slice within a specific HDF5 file.
    - It preprocesses the slices to be compatible with PyTorch transforms.
    """

    def __init__(self, h5_file_list, domain_key_prefix, transform=None):
        """
        Args:
            h5_file_list (List[str]): List of HDF5 file paths for the data split.
            domain_key_prefix (str): The prefix of the dataset to extract (e.g., 'Axial_FLAIR').
            transform (callable, optional): Transformations to apply to each image slice.
        """
        super().__init__(transform=transform)#, cache_size=cache_size)
        self.h5_files = h5_file_list
        self.domain_key_prefix = domain_key_prefix
        self.index_map = []

        print(f"Building index map for domain prefix '{self.domain_key_prefix}'...")

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # 1. Find the first full key name that starts with the provided prefix.
                    full_key_name = next((key for key in f if key.startswith(self.domain_key_prefix)), None)

                    if full_key_name is None:
                        print(
                            f"Warning: No key starting with '{self.domain_key_prefix}' found in {h5_path}. Skipping file.")
                        continue

                    num_slices = f[full_key_name].shape[0]
                    for slice_idx in range(num_slices):
                        # 2. Store the full key we found, not the prefix.
                        self.index_map.append((h5_path, full_key_name, slice_idx))

            except Exception as e:
                print(f"Warning: Could not read or process {h5_path}. Skipping. Error: {e}")
        print(f"Index map built. Total slices: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. Use the map to find file, the full key, and the slice index.
        h5_path, full_key, slice_idx = self.index_map[idx]

        # 2. Load the slice using the full key name.
        with h5py.File(h5_path, 'r') as h5_file_handle:  # Using default cache
            image_slice = h5_file_handle[full_key][slice_idx]

        # 3. Apply Transforms
        image = self._preprocess_slice(image_slice)
        if self.transform:
            image = self.transform(image)
        return image


class HDF5MultiDomainDataset(BaseHDF5Dataset):
    """
    Generalized dataset for loading PAIRED 2D MRI slices from HDF5 files.
    Handles any number of domains, uses prefix matching, and is optimized for performance.
    """

    def __init__(self, h5_file_list, domain_key_prefixes, transform=None):
        """
        Args:
            h5_file_list (List[str]): List of HDF5 file paths for the data split.
            domain_key_prefixes (List[str]): A list of one or more prefixes for the datasets to use.
            transform (callable, optional): Transformations to apply to each slice.
        """
        super().__init__(transform=transform)

        if not domain_key_prefixes:
            raise ValueError("domain_key_prefixes must be a non-empty list of strings.")

        self.domain_key_prefixes = list(domain_key_prefixes)
        self.index_map = []

        print(colored(f"\t\t\tBuilding index map for domains: {self.domain_key_prefixes}", 'green'))
        for h5_path in h5_file_list:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Find the full key name for each domain prefix
                    full_keys = [next((key for key in f if key.startswith(prefix)), None) for prefix in
                                 self.domain_key_prefixes]

                    # Only use this file if ALL domains are present
                    if all(full_keys):
                        # Use the first key to determine the number of slices
                        num_slices = f[full_keys[0]].shape[0]
                        for slice_idx in range(num_slices):
                            # Store the path, a tuple of all full keys, and the slice index
                            self.index_map.append((h5_path, tuple(full_keys), slice_idx))
                    else:
                        print(colored(f"\t\t\tWarning: File {h5_path} is missing one or more required domains. Skipping."), 'yellow')
            except Exception as e:
                print(colored(f"\t\t\tWarning: Could not read or process {h5_path}. Skipping. Error: {e}", 'yellow'))
        print(colored(f"\t\t\tIndex map built. Total slice sets: {len(self.index_map)}", 'green'))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        h5_path, full_keys, slice_idx = self.index_map[idx]

        # Instantiate a file handle
        # 1. Create a dictionary of all the preprocessed NumPy slices
        numpy_dict = {}
        #with h5py.File(h5_path, 'r') as h5_file_handle:
        with h5py.File(h5_path, 'r', swmr=True, libver='latest') as h5_file_handle:
            output_dict = {}
            # Loop through the prefixes and full keys to load and process each slice
            for prefix, full_key in zip(self.domain_key_prefixes, full_keys):
                # OLD: Keeping in case I mess this up
                #slice_data = h5_file_handle[full_key][slice_idx]
                #numpy_dict[prefix] = self._preprocess_slice(slice_data)
                dataset_handle = h5_file_handle[full_key]
                slice_data = dataset_handle[slice_idx]

                # Are we using global min-maxing?
                v_min = dataset_handle.attrs.get('volume_min')
                v_max = dataset_handle.attrs.get('volume_max')

                stats = None
                if v_min is not None and v_max is not None:
                    stats = (float(v_min), float(v_max))

                # Pass stats to the preprocess function
                numpy_dict[prefix] = self._preprocess_slice(slice_data, global_stats=stats)

        # 2. Apply custom transforms (if they exist) only ONCE to the entire dictionary
        if self.transform:
            # All custom transforms return a tensor_dict[prefix]{'input': ..., 'output': ...}
            return self.transform(numpy_dict)
        else:
            # Handle the case where there is no custom transform, ensuring a consistent output format
            output_dict = {}
            to_tensor = T.ToTensor()
            for prefix, np_img in numpy_dict.items():
                tensor = to_tensor(np_img)
                output_dict[prefix] = {'input': tensor, 'output': tensor.clone()}
            return output_dict

class HDF52DSingleDomainDataset(BaseHDF5Dataset):
    """
    Dataset for loading 2D MRI slices for a SINGLE DOMAIN from HDF5 files.
    - It maps a linear index to a specific slice within a specific HDF5 file.
    - It preprocesses the slices to be compatible with PyTorch transforms.
    """

    def __init__(self, h5_file_list, domain_key_prefixes, transform=None, cache_size=1024**2*10):
        """
        Args:
            h5_file_list (List[str]): List of HDF5 file paths for the data split.
            domain_key_prefixes (List[str]): A list of one or more prefixes for the datasets to use.
            transform (callable, optional): Transformations to apply to each slice.
            cache_size (int): The chunk cache size in bytes for each opened file.
        """
        super().__init__(transform=transform, cache_size=cache_size)
        self.h5_files = h5_file_list
        self.domain_key_prefix = domain_key_prefix
        self.transform = transform

        print(f"Building index map for domain prefix '{self.domain_key_prefix}'...")
        self.index_map = []
        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # 1. Find the first full key name that starts with the provided prefix.
                    full_key_name = next((key for key in f if key.startswith(self.domain_key_prefix)), None)

                    if full_key_name is None:
                        print(
                            f"Warning: No key starting with '{self.domain_key_prefix}' found in {h5_path}. Skipping file.")
                        continue

                    num_slices = f[full_key_name].shape[0]
                    for slice_idx in range(num_slices):
                        # 2. Store the full key we found, not the prefix.
                        self.index_map.append((h5_path, full_key_name, slice_idx))

            except Exception as e:
                print(f"Warning: Could not read or process {h5_path}. Skipping. Error: {e}")
        print(f"Index map built. Total slices: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def _preprocess_slice(self, image_slice):
        # If grayscale (H, W), convert to RGB (H, W, 3)
        if image_slice.ndim == 2:
            image_slice = np.stack([image_slice] * 3, axis=-1)

        # Normalize depending on dtype
        if image_slice.dtype in [np.float32, np.float64]:
            max_val = np.max(image_slice)
            if max_val > 0:
                image_slice = image_slice / (max_val + 1e-8) # Add some laplace smoothing
        else:
            image_slice = image_slice.astype(np.float32) / np.iinfo(image_slice.dtype).max
        return image_slice

    def __getitem__(self, idx):
        # 1. Use the map to find file, the full key, and the slice index.
        h5_path, full_key, slice_idx = self.index_map[idx]

        # 2. Load the slice using the full key name.
        with h5py.File(h5_path, 'r') as f:
            image_slice = f[full_key][slice_idx]

        # 3. Apply Transforms
        image = self._preprocess_slice(image_slice)
        if self.transform:
            image = self.transform(image)
        return image
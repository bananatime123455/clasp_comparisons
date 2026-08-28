import h5py
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Maps modality display names → HDF5 dataset key suffixes.
# Must stay in sync with CKPT_TO_HDF5 in comparison_pipeline.py / Clasp_2D_main.py.
CKPT_TO_HDF5 = {
    'Axial_T1_post': 't1ce_transformed',
    'Axial_T1_pre':  't1_transformed',
    'Axial_T2':      't2_transformed',
    'Axial_FLAIR':   'flair_transformed',
}


def resolve_h5_key(modality_name: str) -> str:
    """
    Convert a display-name modality (e.g. 'Axial_T2') to the HDF5 key suffix
    (e.g. 't2_transformed').  Raises ValueError for unknown names.
    """
    key = CKPT_TO_HDF5.get(modality_name)
    if key is None:
        raise ValueError(
            f"Unknown modality {modality_name!r}. "
            f"Valid options: {list(CKPT_TO_HDF5.keys())}"
        )
    return key


def load_h5_slices(h5_file_list, h5_key_1: str, h5_key_2: str):
    """
    Scan a list of H5 files and return two flat lists of (file, key, slice_idx)
    tuples — one per modality — keeping only files that contain both modalities.

    Parameters
    ----------
    h5_file_list : list[Path | str]
    h5_key_1     : HDF5 key suffix for domain A (e.g. 't1ce_transformed')
    h5_key_2     : HDF5 key suffix for domain B (e.g. 't2_transformed')
    """
    index_map_1 = []
    index_map_2 = []

    for h5_path in h5_file_list:
        h5_path = str(h5_path)
        try:
            with h5py.File(h5_path, 'r') as f:
                keys = list(f.keys())
                # Match by suffix — the full stored key may have a leading prefix
                # (e.g. "Axial_T2_t2_transformed"), so we search for containment
                # as well as exact equality to be robust.
                full_key_1 = next(
                    (k for k in keys if k == h5_key_1 or k.endswith(h5_key_1)), None)
                full_key_2 = next(
                    (k for k in keys if k == h5_key_2 or k.endswith(h5_key_2)), None)

                if full_key_1 is None or full_key_2 is None:
                    missing = []
                    if full_key_1 is None:
                        missing.append(h5_key_1)
                    if full_key_2 is None:
                        missing.append(h5_key_2)
                    print(
                        f"Warning: {h5_path} is missing key(s) {missing} "
                        f"(available: {keys}) — skipping."
                    )
                    continue

                n_slices = min(f[full_key_1].shape[0], f[full_key_2].shape[0])
                for i in range(n_slices):
                    index_map_1.append((h5_path, full_key_1, i))
                    index_map_2.append((h5_path, full_key_2, i))

        except Exception as e:
            print(f"Warning: Could not read {h5_path} — skipping. Error: {e}")

    return index_map_1, index_map_2


def _read_and_normalize_slice(h5_path, key, slice_idx):
    """
    Read a single slice from an H5 file and min-max normalize it directly
    to [-1, 1].
    """
    with h5py.File(h5_path, 'r') as f:
        dataset = f[key]
        raw = dataset[slice_idx].astype(np.float32)
 
        # Use volume-level stats if stored, otherwise fall back to slice stats
        v_min = dataset.attrs.get('volume_min')
        v_max = dataset.attrs.get('volume_max')
        if v_min is not None and v_max is not None:
            img_min, img_max = float(v_min), float(v_max)
        else:
            img_min, img_max = float(np.min(raw)), float(np.max(raw))
 
    normalized = 2.0 * (raw - img_min) / (img_max - img_min + 1e-8) - 1.0  # [-1, 1]
    return normalized



def _build_tf_dataset(index_map, img_height, img_width, batch_size, buffer_size, shuffle=True):
    """
    Build a tf.data.Dataset from a flat index_map of (h5_path, key, slice_idx).
    Each element is a (img_height, img_width, 3) float32 tensor in [-1, 1].

    The generator/discriminator (mri_gan.py) are hardcoded to accept
    [256, 256, 3] inputs, so single-channel MRI slices are replicated across
    3 channels rather than changing the model.
    """
    h5_paths   = [entry[0] for entry in index_map]
    keys       = [entry[1] for entry in index_map]
    slice_idxs = [entry[2] for entry in index_map]

    combined = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(h5_paths),
        tf.data.Dataset.from_tensor_slices(keys),
        tf.data.Dataset.from_tensor_slices(slice_idxs),
    ))

    def load_slice(h5_path_t, key_t, slice_idx_t):
        def _py_load(h5_path_b, key_b, slice_idx_b):
            h5_path_str = h5_path_b.numpy().decode('utf-8')
            key_str     = key_b.numpy().decode('utf-8')
            idx         = int(slice_idx_b.numpy())
            img = _read_and_normalize_slice(h5_path_str, key_str, idx)

            # Resize to the model's expected spatial size if needed.
            # PIL mode 'F' (32-bit float) handles the [-1, 1] range without
            # clipping, unlike the default uint8 inference from fromarray().
            if img.shape != (img_height, img_width):
                from PIL import Image as PILImage
                pil = PILImage.fromarray(img, mode='F').resize(
                    (img_width, img_height), PILImage.BILINEAR)
                img = np.array(pil, dtype=np.float32)

            # (H, W) -> (H, W, 3): replicate the grayscale slice across
            # channels so it matches the model's fixed RGB input shape.
            img = np.stack([img, img, img], axis=-1)
            return img

        image = tf.py_function(
            func=_py_load,
            inp=[h5_path_t, key_t, slice_idx_t],
            Tout=tf.float32,
        )
        image.set_shape([img_height, img_width, 3])
        return image

    ds = combined.map(load_slice, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def load_datasets(train_files, test_files, val_files,
                  modality_1: str, modality_2: str,
                  img_height: int, img_width: int,
                  batch_size: int, buffer_size: int):
    """
    Build train/test/val tf.data pipelines for two modalities directly from
    HDF5 files, returning (H, W, 3) float32 tensors in [-1, 1] as np-backed
    tensors ready to feed straight into the existing CycleGAN models.
    """
    h5_key_1 = resolve_h5_key(modality_1)
    h5_key_2 = resolve_h5_key(modality_2)

    print(f"Modality mapping: '{modality_1}' → '{h5_key_1}' | '{modality_2}' → '{h5_key_2}'")

    def _make_pair(file_list, shuffle):
        idx1, idx2 = load_h5_slices(file_list, h5_key_1, h5_key_2)
        ds1 = _build_tf_dataset(idx1, img_height, img_width, batch_size, buffer_size, shuffle=shuffle)
        ds2 = _build_tf_dataset(idx2, img_height, img_width, batch_size, buffer_size, shuffle=shuffle)
        return ds1, ds2

    tr1_train, tr2_train = _make_pair(train_files, shuffle=True)
    tr1_test,  tr2_test  = _make_pair(test_files,  shuffle=False)
    tr1_val,   tr2_val   = _make_pair(val_files,   shuffle=False)

    return tr1_train, tr2_train, tr1_test, tr2_test, tr1_val, tr2_val
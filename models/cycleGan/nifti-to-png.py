import ants
import numpy as np
import pathlib
from PIL import Image
import os

def nifti_to_png(
    input_dir,
    output_dir,
    modality,
):
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all nifti files matching the modality
    all_paths = list(input_dir.rglob("*.nii.gz")) + list(input_dir.rglob("*.nii"))
    filtered = [p for p in all_paths if f"_{modality}." in p.name]

    if not filtered:
        print(f"No files found for modality '{modality}' in {input_dir}")
        print(f"Sample files found: {[p.name for p in all_paths[:5]]}")
        return

    print(f"Found {len(filtered)} files for modality '{modality}'")

    for i, nifti_path in enumerate(filtered):
        try:
            # Load and extract middle slice
            image = ants.image_read(str(nifti_path))
            volume = image.numpy()

            if volume.ndim == 3:
                mid_idx = volume.shape[2] // 2
                slice_np = volume[:, :, mid_idx]
            elif volume.ndim == 2:
                slice_np = volume
            else:
                print(f"Skipping {nifti_path.name} — unexpected shape {volume.shape}")
                continue

            # Normalize to 0–255 for PNG
            slice_np = slice_np.astype(np.float32)
            slice_min, slice_max = slice_np.min(), slice_np.max()
            if slice_max > slice_min:  # avoid division by zero for empty slices
                slice_np = (slice_np - slice_min) / (slice_max - slice_min) * 255.0
            slice_uint8 = slice_np.astype(np.uint8)

            # Transpose to match expected orientation
            slice_uint8 = np.flipud(slice_uint8.T)

            # Save as PNG using subject ID as filename
            subject_id = nifti_path.parent.name  # e.g. BraTS2021_01163
            out_path = output_dir / f"{subject_id}_{modality}.png"
            Image.fromarray(slice_uint8, mode='L').save(out_path)

            if i % 50 == 0:
                print(f"  [{i+1}/{len(filtered)}] Saved {out_path.name}")

        except Exception as e:
            print(f"  Error processing {nifti_path.name}: {e}")
            continue

    print(f"\nDone. PNGs saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  type=str, required=True,
                        help="Root folder containing NIfTI files")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Folder to save PNG files")
    parser.add_argument('--modality',   type=str, default="t1ce",
                        help="Modality to extract e.g. t1, t1ce, t2, flair")
    args = parser.parse_args()

    nifti_to_png(args.input_dir, args.output_dir, args.modality)
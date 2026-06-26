import argparse
import os
import zipfile
import shutil
import urllib.request
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Download and process dataset')
    parser.add_argument('--dataset', help='Name of the dataset.', required=True,
                        choices=['afhq_dog2cat',
                                 'animal_faces'])
    parser.add_argument('--data_dir', default='./dataset',
                        help='Directory to save all datasets.')
    args = parser.parse_args()
    return args


def download_with_progress(url, save_path):
    """Download a file with a simple progress indicator."""
    print(f"Downloading from {url}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            percent = min(percent, 100)
            sys.stdout.write(f"\r  {percent}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, save_path, reporthook)
    print()  # newline after progress


def main():
    args = parse_args()

    # Create the dataset directory.
    os.makedirs(args.data_dir, exist_ok=True)

    folder_path = os.path.join(args.data_dir, args.dataset + '_raw')

    if args.dataset == 'afhq_dog2cat':
        # Download AFHQ from Dropbox (same dataset, reliable link)
        url = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
        zip_path = os.path.join(args.data_dir, 'afhq.zip')
        afhq_path = os.path.join(args.data_dir, 'afhq')

        # Download if zip not already present
        if not os.path.exists(zip_path) and not os.path.exists(afhq_path) \
                and not os.path.exists(folder_path):
            print("Downloading the dataset afhq_dog2cat.")
            download_with_progress(url, zip_path)

        # Extract zip
        if not os.path.exists(afhq_path) and not os.path.exists(folder_path):
            print("Extracting the dataset afhq_dog2cat.")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(args.data_dir)
            os.remove(zip_path)
            print("Extraction complete.")

        # Build the expected folder structure from the extracted AFHQ layout:
        #   afhq_dog2cat_raw/
        #     trainA/  <- afhq/train/dog
        #     trainB/  <- afhq/train/cat
        #     testA/   <- afhq/val/dog  (or test/dog if val doesn't exist)
        #     testB/   <- afhq/val/cat
        if not os.path.exists(folder_path):
            print("Building folder structure at: {}".format(folder_path))
            os.makedirs(folder_path, exist_ok=True)

            # Detect whether the split folder is called 'val' or 'test'
            val_split = 'val' if os.path.exists(
                os.path.join(afhq_path, 'val')) else 'test'

            splits = {
                'trainA': os.path.join(afhq_path, 'train', 'dog'),
                'trainB': os.path.join(afhq_path, 'train', 'cat'),
                'testA':  os.path.join(afhq_path, val_split, 'dog'),
                'testB':  os.path.join(afhq_path, val_split, 'cat'),
            }

            for dest_name, src_path in splits.items():
                dest_path = os.path.join(folder_path, dest_name)
                if not os.path.exists(dest_path):
                    if os.path.exists(src_path):
                        shutil.copytree(src_path, dest_path)
                        print("  Copied {} -> {}".format(src_path, dest_path))
                    else:
                        print("  WARNING: source not found: {}".format(src_path))

            print("Dataset ready at: {}".format(folder_path))

    elif args.dataset == 'animal_faces':
        # Original Google Drive ID — may or may not be accessible
        zip_path = os.path.join(args.data_dir, 'animal_faces.zip')
        if not os.path.exists(zip_path) and not os.path.exists(folder_path):
            print("Downloading the dataset animal_faces.")
            try:
                import gdown
                gdown.download(
                    id='1ftr1xWm0VakGlLUWi7-hdAt9W37luQOA',
                    output=zip_path, quiet=False)
            except Exception as e:
                raise RuntimeError(
                    "Failed to download animal_faces dataset.\n"
                    "Original error: {}\n"
                    "Try downloading manually from Google Drive: "
                    "https://drive.google.com/file/d/1ftr1xWm0VakGlLUWi7-hdAt9W37luQOA"
                    .format(e))

        if not os.path.exists(folder_path):
            print("Extracting the dataset animal_faces.")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(folder_path)
            os.remove(zip_path)
            print("Dataset ready at: {}".format(folder_path))

    else:
        raise ValueError('Invalid dataset {}.'.format(args.dataset))


if __name__ == "__main__":
    main()
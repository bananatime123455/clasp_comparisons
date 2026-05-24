import argparse
from comparison_pipeline import comp_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="MRI GAN Comparison Pipeline")
    parser.add_argument('--h5_path',              type=str, default="/mnt/r/User_Workspaces/Eric_Ma/New_Registration_Results/")
    parser.add_argument('--modality_1',           type=str, default="T1_Post")
    parser.add_argument('--modality_2',           type=str, default="T2")
    parser.add_argument('--data_dir_results',     type=str, default="./Results/")
    parser.add_argument('--models_comparing',     type=str, default="Clasp_2D,cycle_GAN")
    parser.add_argument('--epochs',               type=int, default=100)
    # CycleGAN only
    parser.add_argument('--data_dir_png_t1',      type=str, default=None,
                        help="PNG directory for mod1 (required for cycle_GAN)")
    parser.add_argument('--data_dir_png_t2',      type=str, default=None,
                        help="PNG directory for mod2 (required for cycle_GAN)")
    parser.add_argument('--img_height',           type=int, default=256)
    parser.add_argument('--img_width',            type=int, default=256)
    parser.add_argument('--batch_size',           type=int, default=16)
    parser.add_argument('--buffer_size',          type=int, default=1000)
    parser.add_argument('--early_stopping_patience', type=int,   default=5)
    parser.add_argument('--early_stopping_delta',    type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = comp_pipeline(
        h5_path=args.h5_path,
        modality_1=args.modality_1,
        modality_2=args.modality_2,
        data_dir_results=args.data_dir_results,
        models_comparing=args.models_comparing.split(','),
        epochs=args.epochs,
        data_dir_png_t1=args.data_dir_png_t1,
        data_dir_png_t2=args.data_dir_png_t2,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
    )
    pipeline.compare()
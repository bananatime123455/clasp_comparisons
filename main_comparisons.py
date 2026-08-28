import argparse
from comparison_pipeline import comp_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="MRI GAN Comparison Pipeline")
    parser.add_argument('--h5_path',              type=str, default="./Data/BRATS")
    parser.add_argument('--modality_1', type=str, default="Axial_T1_post",
                        help="Valid options: Axial_T1_post, Axial_T1_pre, Axial_T2, Axial_FLAIR")
    parser.add_argument('--modality_2', type=str, default="Axial_T2",
                        help="Valid options: Axial_T1_post, Axial_T1_pre, Axial_T2, Axial_FLAIR")
    parser.add_argument('--data_dir_results',     type=str, default="./Results/")
    parser.add_argument('--models_comparing',     type=str, default="Clasp_2D",
                        help="Valid options: cycle_GAN,Clasp_2D,UNIT,pix2pixHD,MUNIT (Comma-separated list of models to compare with no spaces before or after commas)") 
    #training parameters
    parser.add_argument('--epochs',               type=int, default=2)
    parser.add_argument('--batch_size',           type=int, default=8)
    parser.add_argument('--buffer_size',          type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int,   default=3)
    parser.add_argument('--early_stopping_delta',    type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Args parsed successfully")
    pipeline = comp_pipeline(
        h5_path=args.h5_path,
        modality_1=args.modality_1,
        modality_2=args.modality_2,
        data_dir_results=args.data_dir_results,
        models_comparing=[m.strip() for m in args.models_comparing.split(',') if m.strip()],
        epochs=args.epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
    )
    print("Starting comparison pipeline...")
    pipeline.compare()

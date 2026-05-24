import argparse
from pipeline import MRIGANPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="MRI CycleGAN Training Pipeline")
    parser.add_argument('--data_dir_t1',             type=str, default="./Tr1/")
    parser.add_argument('--data_dir_t2',             type=str, default="./Tr2/")
    parser.add_argument('--data_dir_results',        type=str, default="./Results/")
    parser.add_argument('--img_height',              type=int, default=256)
    parser.add_argument('--img_width',               type=int, default=256)
    parser.add_argument('--batch_size',              type=int, default=16)
    parser.add_argument('--buffer_size',             type=int, default=1000)
    parser.add_argument('--epochs',                  type=int, default=125)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_delta',    type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = MRIGANPipeline(
        data_dir_t1=args.data_dir_t1,
        data_dir_t2=args.data_dir_t2,
        data_dir_results=args.data_dir_results,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
    )
    metrics = pipeline.train()
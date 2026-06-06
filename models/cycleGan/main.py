import argparse
from pipeline import MRIGANPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="MRI CycleGAN Training Pipeline")
    
    parser.add_argument('--batch_size',              type=int, default=32)
    parser.add_argument('--buffer_size',             type=int, default=1000)
    parser.add_argument('--epochs',                  type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_delta',    type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = MRIGANPipeline(
        train_data
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
    )
    metrics = pipeline.train()
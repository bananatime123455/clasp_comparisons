import argparse
import json
from comparison_pipeline import comp_pipeline


_EPILOG = """\
Examples:

  Basic comparison, all models using the same shared training settings:

    python main_comparisons.py \\
        --models_comparing UNIT,MUNIT,pix2pixHD \\
        --epochs 20 --batch_size 8 --early_stopping_patience 3

  Same run, but MUNIT trains with its own epochs/batch_size/patience via
  --model_params (JSON dict keyed by model name; must be single-quoted so
  the shell passes it through as one argument — unquoted JSON gets mangled
  by brace expansion and whitespace splitting):

    python main_comparisons.py \\
        --models_comparing UNIT,MUNIT,pix2pixHD \\
        --epochs 20 --batch_size 8 --early_stopping_patience 3 \\
        --model_params '{"MUNIT": {"epochs": 10, "batch_size": 2, "early_stopping_patience": 5}}'

  UNIT and pix2pixHD above still use the shared --epochs/--batch_size/
  --early_stopping_patience (20 / 8 / 3). MUNIT ignores those and uses
  epochs=10, batch_size=2, early_stopping_patience=5 instead, falling
  back to the shared early_stopping_delta since it wasn't overridden.

  Overriding more than one model at once — each key only needs to set
  the params it wants to change, everything else falls back to shared:

    --model_params '{"MUNIT": {"epochs": 10, "batch_size": 2}, "UNIT": {"early_stopping_patience": 5}}'
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="MRI GAN Comparison Pipeline",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--h5_path',              type=str, default="./Data/BRATS")
    parser.add_argument('--modality_1', type=str, default="Axial_T1_post",
                        help="Valid options: Axial_T1_post, Axial_T1_pre, Axial_T2, Axial_FLAIR")
    parser.add_argument('--modality_2', type=str, default="Axial_T2",
                        help="Valid options: Axial_T1_post, Axial_T1_pre, Axial_T2, Axial_FLAIR")
    parser.add_argument('--data_dir_results',     type=str, default="./Results/")
    parser.add_argument('--models_comparing',     type=str, default="Clasp_2D",
                        help="Valid options: cycle_GAN,Clasp_2D,UNIT,pix2pixHD,MUNIT (Comma-separated list of models to compare with no spaces before or after commas)") 
    #training parameters (shared defaults — used for any model that doesn't
    #have its own override in --model_params)
    parser.add_argument('--epochs',               type=int, default=100)
    parser.add_argument('--batch_size',           type=int, default=8)
    parser.add_argument('--buffer_size',          type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int,   default=3)
    parser.add_argument('--early_stopping_delta',    type=float, default=0.01)
    #per-model overrides — see the worked examples in --help (epilog above).
    #any key/model omitted falls back to the shared args above.
    parser.add_argument('--model_params', type=str, default=None,
                        help='JSON dict of per-model training/early-stopping '
                             'overrides, keyed by model name (Clasp_2D, '
                             'cycle_GAN, UNIT, MUNIT, pix2pixHD). Each '
                             'model entry may set any of: epochs, '
                             'batch_size, buffer_size, '
                             'early_stopping_patience, early_stopping_delta '
                             '(and, for MUNIT only, munit_batch_size_scale). '
                             'Anything not specified uses the shared '
                             'defaults above. See examples below. Must be '
                             'single-quoted on the command line.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Args parsed successfully")

    if args.model_params:
        try:
            model_params = json.loads(args.model_params)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--model_params is not valid JSON: {e}")
        if not isinstance(model_params, dict):
            raise SystemExit("--model_params must be a JSON object mapping "
                             "model name -> overrides dict.")
    else:
        model_params = {}

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
        model_params=model_params,
    )
    print("Starting comparison pipeline...")
    pipeline.compare()
import argparse
import numpy as np
import pandas as pd

from config import config

from training.metrics import compute_metrics_impl


def read_single_column_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] != 1:
        raise ValueError(f"{path} must have exactly one column, found {df.shape[1]}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate regression predictions against ground truth"
    )
    parser.add_argument("--pred", required=True, help="CSV with predictions")
    parser.add_argument("--gt", required=True, help="CSV with ground truth")
    parser.add_argument("--name", required=True, help="Label to prefix metrics")
    parser.add_argument("--small-dataset", action="store_true", default=False)
    parser.add_argument("--medium-dataset", action="store_true", default=False)
    args = parser.parse_args()

    pred_df = read_single_column_csv(args.pred)
    gt_df = read_single_column_csv(args.gt)

    if args.small_dataset and args.medium_dataset:
        raise ValueError("Use only one of --small-dataset or --medium-dataset")

    if args.small_dataset:
        if len(pred_df) < 200 or len(gt_df) < 200:
            raise ValueError(
                "Small dataset requires at least 200 rows in pred and gt"
            )
        rng = np.random.RandomState(config.RANDOM_SEED)
        selected_idx = np.sort(rng.choice(len(pred_df), size=200, replace=False))
        pred_df = pred_df.iloc[selected_idx].copy()
        gt_df = gt_df.iloc[selected_idx].copy()
    elif args.medium_dataset:
        if len(pred_df) < 4000 or len(gt_df) < 4000:
            raise ValueError(
                "Medium dataset requires at least 4000 rows in pred and gt"
            )
        rng = np.random.RandomState(config.RANDOM_SEED)
        selected_idx = np.sort(rng.choice(len(pred_df), size=4000, replace=False))
        pred_df = pred_df.iloc[selected_idx].copy()
        gt_df = gt_df.iloc[selected_idx].copy()

    if len(pred_df) != len(gt_df):
        raise ValueError(
            "Prediction and ground truth must have the same number of rows: "
            f"{len(pred_df)} != {len(gt_df)}"
        )

    metrics = compute_metrics_impl(pred_df.iloc[:, 0], gt_df.iloc[:, 0])
    print(
        f"{args.name} Metrics - RMSE: {metrics['rmse']:.4f}, "
        f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()

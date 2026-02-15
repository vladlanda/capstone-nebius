import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from config import config


try:
    import preprocess
except ImportError as e:
    raise ImportError(f"Error importing preprocess.py: {e}")


def load_artifacts(model_name: str):
    model_path = os.path.join(
        config.MODEL_PATH, f"{config.VERSION_NAME}_{model_name}.joblib"
    )
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    return joblib.load(model_path)


def load_schema_columns(model_name: str) -> list[str]:
    schema_path = os.path.join(
        config.MODEL_PATH, f"{config.VERSION_NAME}_{model_name}_schema.json"
    )
    with open(schema_path) as f:
        return json.load(f)["columns"]


def read_input_csv(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path, engine="python")


def write_output_csv(output_df: pd.DataFrame, output_path: str) -> None:
    output_df.to_csv(output_path, index=False)


def predict_df(pipeline, input_df: pd.DataFrame) -> pd.DataFrame:
    predictions = pipeline.predict(input_df)
    predictions = np.minimum(predictions, 5)
    return pd.DataFrame(predictions, columns=["review_scores_rating"])


def apply_model(
    input_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    preprocess_args = argparse.Namespace(
        model=model_name,
        seed=config.RANDOM_SEED,
    )
    processed_df = preprocess.preprocess_v2(input_df.copy(), preprocess_args)
    if processed_df is None:
        raise RuntimeError("Preprocessing returned None.")

    if processed_df.shape[0] != input_df.shape[0]:
        raise RuntimeError("Row count changed during preprocessing.")

    expected_cols = load_schema_columns(model_name)
    processed_df = processed_df.reindex(columns=expected_cols, fill_value=0)

    pipeline = load_artifacts(model_name)
    if pipeline is None:
        raise FileNotFoundError(
            f"'{model_name}' model not found. Run 'invoke train_{model_name}' first."
        )

    return predict_df(pipeline, processed_df)


def run_prediction(
    input_path: str,
    output_path: str,
    model_name: str,
    small_dataset: bool,
    medium_dataset: bool,
) -> None:
    input_df = read_input_csv(input_path)
    if small_dataset and medium_dataset:
        raise ValueError("Use only one of --small-dataset or --medium-dataset")
    if small_dataset:
        if len(input_df) < 200:
            raise ValueError(
                f"Small dataset requires at least 200 rows, found {len(input_df)}"
            )
        rng = np.random.RandomState(config.RANDOM_SEED)
        selected_idx = np.sort(rng.choice(len(input_df), size=200, replace=False))
        input_df = input_df.iloc[selected_idx].copy()
    elif medium_dataset:
        if len(input_df) < 4000:
            raise ValueError(
                f"Medium dataset requires at least 4000 rows, found {len(input_df)}"
            )
        rng = np.random.RandomState(config.RANDOM_SEED)
        selected_idx = np.sort(rng.choice(len(input_df), size=4000, replace=False))
        input_df = input_df.iloc[selected_idx].copy()
    output_df = apply_model(
        input_df,
        model_name,
    )
    write_output_csv(output_df, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch predictions with a trained AirBnB model"
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--small-dataset", action="store_true", default=False)
    parser.add_argument("--medium-dataset", action="store_true", default=False)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_prediction(
            args.input,
            args.output,
            "catboost",
            args.small_dataset,
            args.medium_dataset,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    main()

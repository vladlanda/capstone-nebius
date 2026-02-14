import argparse
import json
import os
import sys

import joblib
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


def apply_model(
    input_df: pd.DataFrame,
    model_name: str,
    handle_outliers: bool,
) -> pd.DataFrame:
    preprocess_args = argparse.Namespace(
        model=model_name,
        handle_outliers=handle_outliers,
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

    predictions = pipeline.predict(processed_df)
    return pd.DataFrame(predictions, columns=["review_scores_rating"])


def run_prediction(
    input_path: str,
    output_path: str,
    model_name: str,
    handle_outliers: bool,
) -> None:
    input_df = read_input_csv(input_path)
    output_df = apply_model(
        input_df,
        model_name,
        handle_outliers,
    )
    write_output_csv(output_df, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch predictions with a trained AirBnB model"
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["linear", "ridge", "random_forest", "xgboost"],
        help="Model name to load",
    )
    parser.add_argument("--handle-outliers", action="store_true", default=False)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_prediction(
            args.input,
            args.output,
            args.model,
            args.handle_outliers,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    main()

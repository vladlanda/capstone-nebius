from pathlib import Path

import pandas as pd

from config import config


def _get_results_dir():
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_predictions_impl(y_true, y_pred, output_path, label):
    results = pd.DataFrame({
        'y_true': y_true.values.flatten(),
        'y_pred': y_pred.flatten(),
        'residual': y_true.values.flatten() - y_pred.flatten()
    })
    results.to_csv(output_path, index=False)
    print(f"{label} predictions saved to: {output_path}")
    return output_path


def save_predictions(y_train, y_train_pred, y_test, y_test_pred, version_name, model_name):
    results_dir = _get_results_dir()

    train_results_path = results_dir / f"{version_name}_{model_name}_train_predictions.csv"
    test_results_path = results_dir / f"{version_name}_{model_name}_test_predictions.csv"

    save_predictions_impl(y_train, y_train_pred, train_results_path, "Train")
    save_predictions_impl(y_test, y_test_pred, test_results_path, "Test")

    return train_results_path, test_results_path

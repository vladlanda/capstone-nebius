from pathlib import Path

import pandas as pd

from config import config


def _get_results_dir():
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_predictions(y_train, y_train_pred, y_test, y_test_pred, version_name, model_name):
    results_dir = _get_results_dir()

    train_results = pd.DataFrame({
        'y_true': y_train.values.flatten(),
        'y_pred': y_train_pred.flatten(),
        'residual': y_train.values.flatten() - y_train_pred.flatten()
    })
    train_results_path = results_dir / f"{version_name}_{model_name}_train_predictions.csv"
    train_results.to_csv(train_results_path, index=False)
    print(f"Train predictions saved to: {train_results_path}")

    test_results = pd.DataFrame({
        'y_true': y_test.values.flatten(),
        'y_pred': y_test_pred.flatten(),
        'residual': y_test.values.flatten() - y_test_pred.flatten()
    })
    test_results_path = results_dir / f"{version_name}_{model_name}_test_predictions.csv"
    test_results.to_csv(test_results_path, index=False)
    print(f"Test predictions saved to: {test_results_path}")

    return train_results_path, test_results_path

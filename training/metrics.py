from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from config import config


def compute_metrics(y_train, y_train_pred, y_test, y_test_pred):
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(
        f"\nTrain Metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}"
    )
    print(
        f"Test Metrics - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}"
    )

    return {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }


def _get_results_dir():
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_metrics_summary(metrics, version_name, model_name):
    results_dir = _get_results_dir()
    metrics_summary = pd.DataFrame({
        'metric': ['rmse', 'mae', 'r2'],
        'train': [metrics["train_rmse"], metrics["train_mae"], metrics["train_r2"]],
        'test': [metrics["test_rmse"], metrics["test_mae"], metrics["test_r2"]]
    })
    metrics_path = results_dir / f"{version_name}_{model_name}_metrics.csv"
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    return metrics_path

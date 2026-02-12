from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

from config import config
from training.model import load_catboost_best_params


def find_best_catboost_params(get_data, n_trials=5):
    X_train, X_test, y_train, y_test = get_data()
    base_params = load_catboost_best_params()
    if base_params is None:
        base_params = {
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "rsm": 0.8,
        }

    def _clamp(val, low, high):
        return max(low, min(high, val))

    def objective(trial):
        from catboost import CatBoostRegressor

        params = {
            "iterations": trial.suggest_int(
                "iterations",
                _clamp(int(base_params["iterations"] * 0.7), 300, 1500),
                _clamp(int(base_params["iterations"] * 1.3), 300, 1500),
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                _clamp(base_params["learning_rate"] * 0.7, 0.01, 0.1),
                _clamp(base_params["learning_rate"] * 1.3, 0.01, 0.1),
            ),
            "depth": trial.suggest_int(
                "depth",
                _clamp(base_params["depth"] - 2, 4, 10),
                _clamp(base_params["depth"] + 2, 4, 10),
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                _clamp(base_params["l2_leaf_reg"] * 0.5, 1.0, 10.0),
                _clamp(base_params["l2_leaf_reg"] * 2.0, 1.0, 10.0),
            ),
            "subsample": trial.suggest_float(
                "subsample",
                _clamp(base_params["subsample"] * 0.8, 0.6, 1.0),
                _clamp(base_params["subsample"] * 1.2, 0.6, 1.0),
            ),
            "rsm": trial.suggest_float(
                "rsm",
                _clamp(base_params["rsm"] * 0.8, 0.5, 1.0),
                _clamp(base_params["rsm"] * 1.2, 0.5, 1.0),
            ),
            "loss_function": "RMSE",
            "verbose": False,
            "random_seed": 42,
        }

        cb = CatBoostRegressor(**params)
        cb.fit(X_train, y_train.values.ravel())

        test_pred = cb.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        return test_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    params_path = Path(config.MODEL_PATH) / "catboost_best_params.json"
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    import json

    params_path.write_text(json.dumps(best_params, indent=2))

    return None

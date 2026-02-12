from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from config import config


class MixtureOfExpertsRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        classifier,
        regressor_low,
        regressor_high,
        threshold=90,
        use_soft_gate=True,
        target_min=0,
        target_max=5,
    ):
        self.classifier = classifier
        self.regressor_low = regressor_low
        self.regressor_high = regressor_high
        self.threshold = threshold
        self.use_soft_gate = use_soft_gate
        self.target_min = target_min
        self.target_max = target_max

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y_values = y.values.ravel()
        elif isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y.ravel() if len(y.shape) > 1 else y

        y_binary = (y_values >= self.threshold).astype(int)

        print(f"\n  Training classifier (threshold={self.threshold})...")
        print(f"    Low class: {(y_binary == 0).sum()} samples")
        print(f"    High class: {(y_binary == 1).sum()} samples")
        self.classifier.fit(X, y_binary)

        mask_low = y_binary == 0
        mask_high = y_binary == 1

        X_low, y_low = X[mask_low], y_values[mask_low]
        X_high, y_high = X[mask_high], y_values[mask_high]

        print(f"\n  Training low-rating regressor on {len(X_low)} samples...")
        self.regressor_low.fit(X_low, y_low)

        print(f"  Training high-rating regressor on {len(X_high)} samples...")
        self.regressor_high.fit(X_high, y_high)

        return self

    def predict(self, X):
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
            p_high = probs[:, 1]
        else:
            p_high = self.classifier.predict(X).astype(float)

        y_low = self.regressor_low.predict(X).ravel()
        y_high = self.regressor_high.predict(X).ravel()

        if self.use_soft_gate:
            predictions = (1 - p_high) * y_low + p_high * y_high
        else:
            predictions = np.where(p_high > 0.5, y_high, y_low)

        return np.clip(predictions, self.target_min, self.target_max)

    def predict_proba(self, X):
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        preds = self.classifier.predict(X)
        return np.column_stack([1 - preds, preds])


class MixtureOfExpertsPipeline(BaseEstimator, RegressorMixin):
    def __init__(self, moe_model, scaler=None):
        self.moe_model = moe_model
        self.scaler = scaler if scaler is not None else StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.moe_model.fit(X_scaled, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, "is_fitted_"):
            raise ValueError("MoE pipeline is not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.moe_model.predict(X_scaled)


def _default_moe_params():
    return {
        "threshold_method": "percentile",
        "threshold_percentile": 20,
        "threshold_value": 90,
        "use_soft_gate": True,
        "target_min": 0,
        "target_max": 5,
        "classifier_params": {
            "iterations": 300,
            "learning_rate": 0.1,
            "depth": 6,
            "loss_function": "Logloss",
            "random_seed": 42,
        },
        "regressor_params": {
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "rsm": 0.8,
            "random_seed": 42,
        },
    }


def _load_best_params():
    params_path = Path(config.MODEL_PATH) / "best_params.json"
    if params_path.exists():
        import json

        return json.loads(params_path.read_text())
    return None


def _write_best_params(params):
    params_path = Path(config.MODEL_PATH) / "best_params.json"
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    import json

    params_path.write_text(json.dumps(params, indent=2))


def calculate_threshold(y_train, method="median", percentile=50, fixed_value=90):
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_values = y_train.values.ravel()
    else:
        y_values = y_train

    if method == "median":
        return float(np.median(y_values))
    if method == "mean":
        return float(np.mean(y_values))
    if method == "percentile":
        return float(np.percentile(y_values, percentile))
    if method == "fixed":
        return float(fixed_value)
    raise ValueError(f"Unknown threshold method: {method}")


def load_mixture_of_experts_model(y_train):
    from catboost import CatBoostClassifier, CatBoostRegressor

    params = _load_best_params()
    if params is None:
        params = _default_moe_params()
        _write_best_params(params)

    threshold = calculate_threshold(
        y_train,
        method=params["threshold_method"],
        percentile=params["threshold_percentile"],
        fixed_value=params["threshold_value"],
    )
    classifier = CatBoostClassifier(
        **params["classifier_params"],
        verbose=False,
    )
    regressor_low = CatBoostRegressor(
        **params["regressor_params"],
        loss_function="RMSE",
        verbose=False,
    )
    regressor_high = CatBoostRegressor(
        **params["regressor_params"],
        loss_function="RMSE",
        verbose=False,
    )

    moe = MixtureOfExpertsRegressor(
        classifier=classifier,
        regressor_low=regressor_low,
        regressor_high=regressor_high,
        threshold=threshold,
        use_soft_gate=params["use_soft_gate"],
        target_min=params["target_min"],
        target_max=params["target_max"],
    )

    return MixtureOfExpertsPipeline(moe)

    raise ValueError(f"Unknown model: {model_name}")

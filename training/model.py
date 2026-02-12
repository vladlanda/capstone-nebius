from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import config


def load_catboost_best_params():
    params_path = Path(config.MODEL_PATH) / "catboost_best_params.json"
    if params_path.exists():
        import json

        return json.loads(params_path.read_text())
    return None


def _build_pipeline(model, use_scaler=True):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def load_linear_model():
    from sklearn.linear_model import LinearRegression

    return _build_pipeline(LinearRegression())


def load_ridge_model():
    from sklearn.linear_model import Ridge

    return _build_pipeline(Ridge(alpha=1.0))


def load_random_forest_model():
    from sklearn.ensemble import RandomForestRegressor

    return _build_pipeline(RandomForestRegressor(n_estimators=200))


def load_xgboost_model():
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
    )
    return _build_pipeline(model)


def load_catboost_model():
    from catboost import CatBoostRegressor

    catboost_params = load_catboost_best_params()
    if catboost_params is None:
        raise ValueError("catboost_best_params.json not found; run Optuna tuning first")
    return CatBoostRegressor(
        **catboost_params,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )


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


def get_classifier(classifier_name: str, **hyperparams):
    random_state = hyperparams.get("random_state", 42)

    if classifier_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=random_state, max_iter=1000)

    if classifier_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=hyperparams.get("clf_n_estimators", 100),
            learning_rate=hyperparams.get("clf_learning_rate", 0.1),
            max_depth=hyperparams.get("clf_max_depth", 3),
            random_state=random_state,
            n_jobs=-1,
        )

    if classifier_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm") from exc

        return LGBMClassifier(
            n_estimators=hyperparams.get("clf_n_estimators", 100),
            learning_rate=hyperparams.get("clf_learning_rate", 0.1),
            max_depth=hyperparams.get("clf_max_depth", 3),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if classifier_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=hyperparams.get("clf_n_estimators", 100),
            max_depth=hyperparams.get("clf_max_depth", None),
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier: {classifier_name}")


def get_regressor(regressor_name: str, **hyperparams):
    random_state = hyperparams.get("random_state", 42)

    if regressor_name == "linear":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    if regressor_name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=hyperparams.get("reg_alpha", 1.0))

    if regressor_name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(alpha=hyperparams.get("reg_alpha", 1.0), max_iter=10000)

    if regressor_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=hyperparams.get("reg_n_estimators", 200),
            max_depth=hyperparams.get("reg_max_depth", None),
            min_samples_split=hyperparams.get("reg_min_samples_split", 2),
            random_state=random_state,
            n_jobs=-1,
        )

    if regressor_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=hyperparams.get("reg_n_estimators", 300),
            learning_rate=hyperparams.get("reg_learning_rate", 0.05),
            max_depth=hyperparams.get("reg_max_depth", 6),
            random_state=random_state,
            n_jobs=-1,
        )

    if regressor_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm") from exc

        return LGBMRegressor(
            n_estimators=hyperparams.get("reg_n_estimators", 300),
            learning_rate=hyperparams.get("reg_learning_rate", 0.05),
            max_depth=hyperparams.get("reg_max_depth", 6),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if regressor_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=hyperparams.get("reg_n_estimators", 200),
            learning_rate=hyperparams.get("reg_learning_rate", 0.1),
            max_depth=hyperparams.get("reg_max_depth", 3),
            random_state=random_state,
        )

    raise ValueError(f"Unknown regressor: {regressor_name}")


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


def load_mixture_of_experts_model(
    y_train,
    classifier_name="xgboost",
    regressor_name="xgboost",
    threshold_method="median",
    threshold_percentile=50,
    threshold_value=90,
    use_soft_gate=True,
    target_min=0,
    target_max=5,
    **hyperparams,
):
    threshold = calculate_threshold(
        y_train,
        method=threshold_method,
        percentile=threshold_percentile,
        fixed_value=threshold_value,
    )
    classifier = get_classifier(classifier_name, **hyperparams)
    regressor_low = get_regressor(regressor_name, **hyperparams)
    regressor_high = get_regressor(regressor_name, **hyperparams)

    moe = MixtureOfExpertsRegressor(
        classifier=classifier,
        regressor_low=regressor_low,
        regressor_high=regressor_high,
        threshold=threshold,
        use_soft_gate=use_soft_gate,
        target_min=target_min,
        target_max=target_max,
    )

    return MixtureOfExpertsPipeline(moe)


def get_pipeline(model_name: str):
    if model_name == "linear":
        return load_linear_model()

    if model_name == "ridge":
        return load_ridge_model()

    if model_name == "random_forest":
        return load_random_forest_model()

    if model_name == "xgboost":
        return load_xgboost_model()

    if model_name == "catboost":
        return load_catboost_model()

    raise ValueError(f"Unknown model: {model_name}")
